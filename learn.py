import numpy as np
import sys
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from utils.prioritized_replay_buffer import NaivePrioritizedReplayMemory, Transition
from utils.history_buffer import HistoryBuffer
from utils.general import export_plot


class DRRTrainer(object):
    def __init__(self,
                 config,
                 actor_function,
                 critic_function,
                 state_rep_function,
                 reward_function,
                 users,
                 items,
                 train_data,
                 test_data,
                 user_embeddings,
                 item_embeddings,
                 cuda):

        # Import reward function
        self.reward_function = reward_function

        # Initialize device
        self.device_id = torch.cuda.current_device()
        print("CUDA Device ID: ", self.device_id)
        print(torch.cuda.get_device_name(self.device_id))
        print("CUDA Memory Allocated: ", torch.cuda.memory_allocated(self.device_id))
        print("CUDA Memory Reserved: ", torch.cuda.memory_reserved(self.device_id) / 1000000000, "GB")
        torch.cuda.empty_cache()
        self.device = torch.device('cuda:{}'.format(self.device_id) if cuda else "cpu")
        print("Current PyTorch Device: ", self.device)

        # Import Data
        self.train_data = train_data
        self.test_data = test_data
        self.users = users
        self.items = items
        self.user_embeddings = user_embeddings.to(self.device)
        self.item_embeddings = item_embeddings
        self.u = 2
        self.i = 4
        self.r = 1
        self.ti = 0

        # Dimensions
        self.item_features = self.item_embeddings.shape[1]
        self.user_features = self.user_embeddings.shape[1]
        self.n_items = self.item_embeddings.shape[0]
        self.n_users = self.user_embeddings.shape[0]
        self.state_shape = 3 * self.item_features  # dimensionality 3k for drr-ave
        self.action_shape = self.item_features
        self.critic_output_shape = 1
        self.config = config
        print("Data dimensions extracted")

        # Initialize neural networks
        self.state_rep_net = state_rep_function(self.config.history_buffer_size,
                                                self.item_features,
                                                self.user_features)

        self.actor_net = actor_function(self.state_shape,
                                        self.action_shape)
        self.target_actor_net = actor_function(self.state_shape,
                                               self.action_shape)

        self.critic_net = critic_function(self.action_shape,
                                          self.state_shape,
                                          self.critic_output_shape)
        self.target_critic_net = critic_function(self.action_shape,
                                                 self.state_shape,
                                                 self.critic_output_shape)
        print("Models initialized")

        def init_weights(m):
            if hasattr(m, 'weight'):
                nn.init.orthogonal_(m.weight.data)
            if hasattr(m, 'bias'):
                nn.init.constant_(m.bias.data, 0)

        # Initialize weights
        self.state_rep_net.apply(init_weights)
        self.actor_net.apply(init_weights)
        self.critic_net.apply(init_weights)

        # Copy weights target networks
        self.target_actor_net.load_state_dict(
            self.actor_net.state_dict())
        self.target_critic_net.load_state_dict(
            self.critic_net.state_dict())
        print("Model weights initialized, copied to target")

        # Move models and data to CUDA
        if cuda:
            # models
            self.reward_function.cuda()
            self.state_rep_net.cuda()
            self.actor_net.cuda()
            self.target_actor_net.cuda()
            self.critic_net.cuda()
            self.target_critic_net.cuda()

            print("All models, train data, and user embeddings data moved to CUDA")

        # Init optimizers
        self.state_rep_optimizer = torch.optim.Adam(self.state_rep_net.parameters(), lr=self.config.lr_state_rep,
                                                    betas=(0.9, 0.999), eps=1e-08,
                                                    weight_decay=self.config.weight_decay, amsgrad=False)
        self.actor_optimizer = torch.optim.Adam(self.actor_net.parameters(), lr=self.config.lr_actor,
                                                betas=(0.9, 0.999), eps=1e-08, weight_decay=self.config.weight_decay,
                                                amsgrad=False)
        self.critic_optimizer = torch.optim.Adam(self.critic_net.parameters(), lr=self.config.lr_critic,
                                                 betas=(0.9, 0.999), eps=1e-08, weight_decay=self.config.weight_decay,
                                                 amsgrad=False)
        print("Optimizers initialized")

    def load_parameters(self):
        self.state_rep_net.load_state_dict(torch.load(self.config.state_rep_model_trained))
        self.actor_net.load_state_dict(torch.load(self.config.actor_model_trained))
        self.critic_net.load_state_dict(torch.load(self.config.critic_model_trained))
        self.target_actor_net.load_state_dict(self.actor_net.state_dict())
        self.target_critic_net.load_state_dict(self.critic_net.state_dict())

    def learn(self):
        # Transfer training data to device
        self.train_data = self.train_data.to(self.device)

        # Init buffers
        replay_buffer = NaivePrioritizedReplayMemory(self.config.replay_buffer_size,
                                                     prob_alpha=self.config.prob_alpha)
        history_buffer = HistoryBuffer(self.config.history_buffer_size)

        # Init trackers
        timesteps = epoch = 0
        eps_slope = abs(self.config.eps_start - self.config.eps) / self.config.eps_steps
        eps = self.config.eps_start
        actor_losses = []
        critic_losses = []
        epi_rewards = []
        epi_avg_rewards = []
        e_arr = []

        # Get users, shuffle, andgo through array
        user_idxs = np.array(list(self.users.values()))
        np.random.shuffle(user_idxs)

        # Start episodes
        for idx, e in enumerate(user_idxs):
            # ---------------------------- start of episode ---------------------------- #

            # Stop if > than max
            if timesteps - self.config.learning_start > self.config.max_timesteps_train:
                break

            # Extract positive user reviews from training
            user_reviews = self.train_data[self.train_data[:, self.u] == e]
            pos_user_reviews = user_reviews[user_reviews[:, self.r] > 0]

            # Move on to next user if not enough positive reviews
            if pos_user_reviews.shape[0] < self.config.history_buffer_size:
                continue

            # Copy item embeddings to candidate item embeddings set
            candidate_items = self.item_embeddings.detach().clone().to(self.device)

            # Sort positive user reviews by timestamp
            pos_user_reviews = pos_user_reviews[pos_user_reviews[:, self.ti].sort()[1]]

            # Extract user embedding tensor
            user_emb = self.user_embeddings[e]

            # Fill history buffer with positive user item embeddings and
            # Remove item embeddings from candidate item set
            ignored_items = []
            for i in range(self.config.history_buffer_size):
                emb = candidate_items[pos_user_reviews[i, self.i]]
                history_buffer.push(emb.detach().clone())

            # Initialize rewards tracker
            rewards = []

            # Starting item index
            t = 0

            state = None
            action = None
            reward = None
            next_state = None
            while t < self.config.episode_length:
                # ---------------------------- start of timestep ---------------------------- #

                # observe current state
                # choose action according to actor network or exploration
                if eps > self.config.eps:
                    eps -= eps_slope
                else:
                    eps = self.config.eps
                state = self.state_rep_net(user_emb, torch.stack(history_buffer.to_list()))
                with torch.no_grad():
                    if np.random.uniform(0, 1) < eps:
                        action = torch.from_numpy(0.1 * np.random.rand(self.action_shape)).float().to(self.device)
                    else:
                        action = self.actor_net(state.detach())

                # Calculate ranking scores across items, discard ignored items
                ranking_scores = candidate_items @ action
                rec_items = torch.stack(ignored_items) if len(ignored_items) > 0 else []
                ranking_scores[rec_items] = -float("inf")

                # Get recommended item
                rec_item_idx = torch.argmax(ranking_scores)
                rec_item_emb = candidate_items[rec_item_idx]

                # Get item reward
                if rec_item_idx in user_reviews[:, self.i]:
                    # Reward from rating in dataset if item rated by user
                    reward = user_reviews[user_reviews[:, self.i] == rec_item_idx, self.r][0]
                else:
                    # Item not rated by user, reward from PMF
                    with torch.no_grad():
                        if self.config.zero_reward:
                            reward = torch.tensor(0).to(self.device)
                        else:
                            reward = self.reward_function(torch.tensor(e).to(self.device), rec_item_idx)

                # Track episode rewards
                rewards.append(reward.item())

                # Add item to history buffer if positive review, remove from candidate set
                # Set next state to new or old
                if reward > 0:
                    # Update history buffer with new item
                    history_buffer.push(rec_item_emb.detach().clone())
                    # Observe next state
                    with torch.no_grad():
                        next_state = self.state_rep_net(user_emb, torch.stack(history_buffer.to_list()))
                else:
                    # Keep history buffer the same, next state is current state
                    next_state = state.detach()

                # Remove new item from future recommendations
                ignored_items.append(torch.tensor(rec_item_idx).to(self.device))

                # Add (state, action, reward, next_state) to experience replay
                replay_buffer.push(state,
                                   action,
                                   next_state,
                                   reward
                                   )

                # TRAIN
                if (timesteps > self.config.learning_start) and \
                        (len(replay_buffer) >=
                         self.config.batch_size) and \
                        (timesteps % self.config.learning_freq == 0):

                    critic_loss, actor_loss, critic_params_norm = self.training_step(timesteps,
                                                                                     replay_buffer,
                                                                                     True
                                                                                     )

                    # LOGGING
                    actor_losses.append(actor_loss)
                    critic_losses.append(critic_loss)

                    if timesteps % self.config.log_freq == 0:
                        if len(rewards) > 0:
                            print(
                                f'Timestep {timesteps - self.config.learning_start} | '
                                f'Episode {epoch} | '
                                f'Mean Ep R '
                                f'{np.mean(rewards):.4f} | '
                                f'Max R {np.max(rewards):.4f} | '
                                f'Critic Params Norm {critic_params_norm:.4f} | '
                                f'Actor Loss {actor_loss:.4f} | '
                                f'Critic Loss {critic_loss:.4f} | ')
                            sys.stdout.flush()

                # Housekeeping
                t += 1
                timesteps += 1

                # ---------------------------- end of timestep ---------------------------- #

            # ---------------------------- end of episode ---------------------------- #

            # Logging
            if timesteps - self.config.learning_start > t:
                epoch += 1
                e_arr.append(epoch)
                epi_rewards.append(np.sum(rewards))
                epi_avg_rewards.append(np.mean(rewards))

            if t % self.config.saving_freq == 0:
                export_plot(actor_losses, 'Actor Loss (Training)', self.config.train_actor_loss_plot_dir)
                export_plot(critic_losses, 'Critic Loss (Training)', self.config.train_critic_loss_plot_dir)
                export_plot(epi_avg_rewards,
                            'Average Episodic Reward (Training)',
                            self.config.train_mean_reward_plot_dir)

        print('Training Finished')

        # Save final model parameters
        torch.save(self.actor_net.state_dict(),
                   self.config.actor_model_dir)
        torch.save(self.critic_net.state_dict(),
                   self.config.critic_model_dir)
        torch.save(self.state_rep_net.state_dict(),
                   self.config.state_rep_model_dir)

        # Save data
        np.save(self.config.train_actor_loss_data_dir, actor_losses)
        np.save(self.config.train_critic_loss_data_dir, critic_losses)
        np.save(self.config.train_mean_reward_data_dir, epi_avg_rewards)

        # Export plots
        export_plot(actor_losses, 'Actor Loss (Training)', self.config.train_actor_loss_plot_dir)
        export_plot(critic_losses, 'Critic Loss (Training)', self.config.train_critic_loss_plot_dir)
        export_plot(epi_avg_rewards,
                    'Average Episodic Reward (Training)',
                    self.config.train_mean_reward_plot_dir)

        return actor_losses, critic_losses, epi_avg_rewards

    def training_step(self, t, replay_buffer, training):
        # Create batches
        transitions, indicies, weights = replay_buffer.sample(self.config.batch_size, beta=self.config.beta)
        batch = Transition(*zip(*transitions))
        next_state_batch = torch.cat(batch.next_state).view(
            self.config.batch_size, -1)
        state_batch = torch.cat(batch.state).view(
            self.config.batch_size, -1)
        action_batch = torch.cat(batch.action).view(
            self.config.batch_size, -1)
        reward_batch = torch.stack(batch.reward).view(
            self.config.batch_size, -1)

        # ---------------------------- Update Critic Network ---------------------------- #

        # Calculate Critic loss
        critic_loss, new_priorities = self.compute_prioritized_dqn_loss(state_batch.detach(),
                                                                        action_batch,
                                                                        reward_batch,
                                                                        next_state_batch,
                                                                        weights)

        # Minimize loss, update parameters, update priorities
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        replay_buffer.update_priorities(indicies, new_priorities)
        critic_param_norm = torch.nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.config.clip_val)
        self.critic_optimizer.step()

        # ----------------------------- Update Actor Network ---------------------------- #

        self.actor_optimizer.zero_grad()
        self.state_rep_optimizer.zero_grad()

        # Compute actor loss
        actions_pred = self.actor_net(state_batch)
        actor_loss = -self.critic_net(state_batch.detach(), actions_pred).mean()

        # Minimize the loss
        actor_loss.backward(retain_graph=True)
        self.actor_optimizer.step()
        self.state_rep_optimizer.step()

        # ----------------------- Soft update the target networks ----------------------- #
        self.soft_update(self.critic_net, self.target_critic_net, self.config.tau)
        self.soft_update(self.actor_net, self.target_actor_net, self.config.tau)

        # ---------------------------- Save models at checkpoints ---------------------------- #

        if t % self.config.saving_freq == 0 and training:
            torch.save(self.actor_net.state_dict(),
                       self.config.actor_model_dir)
            torch.save(self.critic_net.state_dict(),
                       self.config.critic_model_dir)
            torch.save(self.state_rep_net.state_dict(),
                       self.config.state_rep_model_dir)

        return critic_loss.item(), actor_loss.item(), critic_param_norm

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def compute_prioritized_dqn_loss(self,
                                     state_batch,
                                     action_batch,
                                     reward_batch,
                                     next_state_batch,
                                     weights):
        '''
        :param state_batch: (torch tensor) shape = (batch_size x state_dims),
                The batched tensor of states collected during
                training (i.e. s)
        :param action_batch: (torch LongTensor) shape = (batch_size,)
                The actions that you actually took at each step (i.e. a)
        :param reward_batch: (torch tensor) shape = (batch_size,)
                The rewards that you actually got at each step (i.e. r)
        :param next_state_batch: (torch tensor) shape = (batch_size x state_dims),
                The batched tensor of next states collected during
                training (i.e. s')
        :param weights: (torch tensor) shape = (batch_size,)
                Weights for each batch item w.r.t. prioritized experience replay buffer
        :return: loss: (torch tensor) shape = (1),
                 new_priorities: (numpy array) shape = (batch_size,)
        '''

        # Extract target network Q values
        with torch.no_grad():
            next_action = self.target_actor_net(next_state_batch)
            q_target = self.target_critic_net(next_state_batch, next_action)

        # Build y
        y = reward_batch + self.config.gamma * q_target

        # Get Q values for current state
        q_vals = self.critic_net(state_batch, action_batch)

        # Calculate loss
        loss = y - q_vals
        loss = loss.flatten()
        loss = loss.pow(2)
        weights_ten = torch.tensor(weights, requires_grad=False).to(self.device)
        loss = loss * weights_ten

        # Calculate new priorities
        new_priorities = (loss + 1e-5).cpu().detach().numpy()
        loss = loss.mean()

        return loss, new_priorities

    def online_evaluate(self):
        # Load model parameters
        self.load_parameters()

        # Get test data ready
        self.test_data = self.test_data.to(self.device)

        # Init buffers
        replay_buffer = NaivePrioritizedReplayMemory(self.config.replay_buffer_size, prob_alpha=self.config.prob_alpha)
        history_buffer = HistoryBuffer(self.config.history_buffer_size)

        # Init trackers
        timesteps = epoch = 0
        actor_losses = []
        critic_losses = []
        rewards = []

        # Get users, shuffle, and go through array
        user_idxs = np.array(list(self.users.values()))
        np.random.shuffle(user_idxs)

        # Start episodes
        for idx, e in enumerate(user_idxs):
            # ---------------------------- start of episode ---------------------------- #

            # Stop if > than max
            if timesteps > self.config.max_timesteps_online:
                break

            # Extract positive user reviews from training
            user_reviews = self.test_data[self.test_data[:, self.u] == e]
            pos_user_reviews = user_reviews[user_reviews[:, self.r] > 0]

            # Move on to next user if not enough positive reviews
            if pos_user_reviews.shape[0] < self.config.history_buffer_size:
                continue

            # Copy item embeddings to candidate item embeddings set
            candidate_items = self.item_embeddings.detach().clone().to(self.device)

            # Sort positive user reviews by timestamp
            pos_user_reviews = pos_user_reviews[pos_user_reviews[:, self.ti].sort()[1]]

            # Extract user embedding tensor
            user_emb = self.user_embeddings[e]

            # Fill history buffer with positive user item embeddings and
            # Remove item embeddings from candidate item set
            ignored_items = []
            for i in range(self.config.history_buffer_size):
                emb = candidate_items[pos_user_reviews[i, self.i]]
                history_buffer.push(emb.detach().clone())

            # Starting item index
            t = 0

            # Reload before each session
            self.load_parameters()

            state = None
            action = None
            reward = None
            next_state = None
            while t < self.config.episode_length:
                # ---------------------------- start of timestep ---------------------------- #

                # observe current state
                # choose action according to actor network or exploration
                state = self.state_rep_net(user_emb, torch.stack(history_buffer.to_list()))
                with torch.no_grad():
                    if np.random.uniform(0, 1) < self.config.eps_eval:
                        action = torch.from_numpy(0.1 * np.random.rand(self.action_shape)).float().to(self.device)
                    else:
                        action = self.actor_net(state.detach())

                # Calculate ranking scores across items, discard ignored items
                ranking_scores = candidate_items @ action
                rec_items = torch.stack(ignored_items) if len(ignored_items) > 0 else []
                ranking_scores[rec_items] = -float("inf")

                # Get recommended item
                rec_item_idx = torch.argmax(ranking_scores)
                rec_item_emb = candidate_items[rec_item_idx]

                # Get item reward
                if rec_item_idx in user_reviews[:, self.i]:
                    # Reward from rating in dataset if item rated by user
                    reward = user_reviews[user_reviews[:, self.i] == rec_item_idx, self.r][0]
                else:
                    # Item not rated by user, reward from PMF
                    with torch.no_grad():
                        reward = self.reward_function(torch.tensor(e).to(self.device), rec_item_idx)

                # Track episode rewards
                rewards.append(reward.item())

                # Add item to history buffer if positive review, remove from candidate set
                # Set next state to new or old
                if reward > 0:
                    # Update history buffer with new item
                    history_buffer.push(rec_item_emb.detach().clone())
                    # Observe next state
                    with torch.no_grad():
                        next_state = self.state_rep_net(user_emb, torch.stack(history_buffer.to_list()))
                else:
                    # Keep history buffer the same, next state is current state
                    next_state = state.detach()

                # Remove new item from future recommendations
                ignored_items.append(torch.tensor(rec_item_idx).to(self.device))

                # Add (state, action, reward, next_state) to experience replay
                replay_buffer.push(state,
                                   action,
                                   next_state,
                                   reward
                                   )

                # TRAIN
                if (len(replay_buffer) >= self.config.batch_size) and \
                        (timesteps % self.config.learning_freq == 0):

                    critic_loss, actor_loss, critic_params_norm = self.training_step(timesteps,
                                                                                     replay_buffer,
                                                                                     False
                                                                                     )

                    # LOGGING
                    actor_losses.append(actor_loss)
                    critic_losses.append(critic_loss)

                    if timesteps % self.config.log_freq == 0:
                        if len(rewards) > 0:
                            print(
                                f'Timestep {timesteps} | '
                                f'Episode {epoch} | '
                                f'Avg Total Reward {np.mean(rewards):.4f} | '
                                f'Critic Params Norm {critic_params_norm:.4f} | '
                                f'Actor Loss {actor_loss:.4f} | '
                                f'Critic Loss {critic_loss:.4f} | ')
                            sys.stdout.flush()

                # Housekeeping
                t += 1
                timesteps += 1

                # ---------------------------- end of timestep ---------------------------- #

            # ---------------------------- end of episode ---------------------------- #

            # Housekeeping
            del candidate_items

            epoch += 1

        print('Online Evaluation Finished')
        print(f'Average Reward {np.mean(rewards):.4f} | ')
        x = np.arange(len(actor_losses))
        plt.plot(x, actor_losses, label="Test Actor")
        plt.plot(x, critic_losses, label="Test Critic")
        plt.legend()
        plt.xlabel('Timestep (t)')
        plt.ylabel('Loss')
        plt.title('Actor and Critic Losses (Evaluation)')
        plt.minorticks_on()

        # Reload model parameters
        self.load_parameters()

        return np.mean(rewards)

    def offline_evaluate(self, T):
        # Load model parameters
        self.load_parameters()

        # Get test data ready
        self.test_data = self.test_data.to(self.device)

        # Init data tracking
        # data_dict = {
        #     'Timestep': 0,
        #     'Training Rewards': 0,
        #     'Loss': 0
        # }
        # fieldnames = [key for key, _ in data_dict.items()]
        # csv_logger = CSVLogger(fieldnames=fieldnames,
        #                        filename=self.config.csv_dir)

        # Init buffers
        history_buffer = HistoryBuffer(self.config.history_buffer_size)

        # Init trackers
        timesteps = epoch = 0
        rewards = []
        epi_precisions = []
        e_arr = []

        # Get users, shuffle, andgo through array
        user_idxs = np.array(list(self.users.values()))
        np.random.shuffle(user_idxs)

        # Start episodes
        for idx, e in enumerate(user_idxs):
            # ---------------------------- start of episode ---------------------------- #

            if len(e_arr) > self.config.max_epochs_offline:
                break

            # Extract user reviews and positive user reviews from test
            user_reviews = self.test_data[self.test_data[:, self.u] == e]
            pos_user_reviews = user_reviews[user_reviews[:, self.r] > 0]

            # Move on to next user if not enough positive or regular reviews
            if user_reviews.shape[0] < T or pos_user_reviews.shape[0] < self.config.history_buffer_size:
                continue

            # Sort user reviews by timestamp
            user_reviews = user_reviews[user_reviews[:, self.ti].sort()[1]]
            pos_user_reviews = pos_user_reviews[pos_user_reviews[:, self.ti].sort()[1]]

            # Copy item embeddings to candidate item embeddings set
            candidate_items = self.item_embeddings.detach().clone().to(self.device)
            user_candidate_items = self.item_embeddings[user_reviews[:, self.i]].detach().clone().to(self.device)

            # Extract user embedding tensor
            user_emb = self.user_embeddings[e]

            # Fill history buffer with positive user item embeddings and
            # Remove item embeddings from candidate item set
            ignored_items = []
            for i in range(self.config.history_buffer_size):
                emb = candidate_items[pos_user_reviews[i, self.i]]
                history_buffer.push(emb.detach().clone())
                # ignored_items.append(pos_user_reviews[i, self.i])

            # Starting item index
            t = 0

            state = None
            action = None
            reward = None
            next_state = None
            while t < T:
                # ---------------------------- start of timestep ---------------------------- #

                # observe current state
                # choose action according to actor network or exploration
                with torch.no_grad():
                    state = self.state_rep_net(user_emb, torch.stack(history_buffer.to_list()))
                    if np.random.uniform(0, 1) < self.config.eps_eval:
                        action = torch.from_numpy(0.1 * np.random.rand(self.action_shape)).float().to(self.device)
                    else:
                        action = self.actor_net(state.detach())

                # Calculate ranking scores across items, discard ignored items
                ranking_scores = candidate_items @ action
                rec_items = torch.stack(ignored_items) if len(ignored_items) > 0 else []
                ranking_scores[rec_items[:, self.i] if len(ignored_items) > 0 else []] = -float("inf")

                # Get recommended item
                rec_item_idx = torch.argmax(ranking_scores[user_reviews[:, self.i]])
                rec_item_emb = user_candidate_items[rec_item_idx]

                # Get item reward
                reward = user_reviews[rec_item_idx, self.r]

                # Track episode rewards
                rewards.append(reward.item())

                # Add item to history buffer if positive review, remove from candidate set
                # Set next state to new or old
                if reward > 0:
                    # Update history buffer with new item
                    history_buffer.push(rec_item_emb.detach().clone())
                    # Observe next state
                    with torch.no_grad():
                        next_state = self.state_rep_net(user_emb, torch.stack(history_buffer.to_list()))
                else:
                    # Keep history buffer the same, next state is current state
                    next_state = state.detach()

                # Remove new item from future recommendations
                ignored_items.append(user_reviews[rec_item_idx])

                # Housekeeping
                t += 1
                timesteps += 1

                # ---------------------------- end of timestep ---------------------------- #

            # ---------------------------- end of episode ---------------------------- #

            # T_indicies = np.arange(T)
            # rel_real = user_reviews[T_indicies]
            # rel_real = rel_real[rel_real[:, self.r] > 0]
            rec_items = torch.stack(ignored_items)
            rel_pred = rec_items[rec_items[:, self.r] > 0]
            precision_T = len(rel_pred) / len(rec_items)

            # Logging
            epoch += 1
            e_arr.append(epoch)
            epi_precisions.append(precision_T)

            if timesteps % self.config.log_freq == 0:
                if len(rewards) > 0:
                    print(f'Episode {epoch} | '
                          f'Precision@{T} {precision_T} | '
                          f'Avg Precision@{T} {np.mean(epi_precisions):.4f} | '
                          )
                    sys.stdout.flush()

        print('Offline Evaluation Finished')
        print(f'Average Precision@{T}: {np.mean(epi_precisions):.4f} | ')
        plt.plot(e_arr, epi_precisions, label=f'Precision@{T}')
        # plt.plot(x, critic_losses, label="Test Critic")
        plt.legend()
        plt.xlabel('Episode (t)')
        plt.ylabel('Precesion@T')
        plt.title('Precision@T (Offline Evaluation)')
        plt.minorticks_on()

        # Reload model parameters
        self.load_parameters()

        return np.mean(epi_precisions)

    def offline_pmf_evaluate(self, T):
        # Load model parameters
        self.load_parameters()

        # Get test data ready
        self.test_data = self.test_data.to(self.device)

        # Init buffers
        history_buffer = HistoryBuffer(self.config.history_buffer_size)

        # Init trackers
        timesteps = epoch = 0
        rewards = []
        epi_precisions = []
        e_arr = []

        # Get users, shuffle, andgo through array
        user_idxs = np.array(list(self.users.values()))
        np.random.shuffle(user_idxs)

        candidate_item_idxs = np.arange(self.item_embeddings.shape[0])
        candidate_item_idxs = torch.from_numpy(candidate_item_idxs).to(self.device).long()

        # Start episodes
        for idx, e in enumerate(user_idxs):
            # ---------------------------- start of episode ---------------------------- #

            if len(e_arr) > self.config.max_epochs_offline:
                break

            # Extract user reviews and positive user reviews from test
            user_reviews = self.test_data[self.test_data[:, self.u] == e]
            pos_user_reviews = user_reviews[user_reviews[:, self.r] > 0]

            # Move on to next user if not enough positive or regular reviews
            if user_reviews.shape[0] < T or pos_user_reviews.shape[0] < self.config.history_buffer_size:
                continue

            # Sort user reviews by timestamp
            user_reviews = user_reviews[user_reviews[:, self.ti].sort()[1]]
            pos_user_reviews = pos_user_reviews[pos_user_reviews[:, self.ti].sort()[1]]

            # Copy item embeddings to candidate item embeddings set
            candidate_items = self.item_embeddings.detach().clone().to(self.device)
            user_candidate_items = self.item_embeddings[user_reviews[:, self.i]].detach().clone().to(self.device)

            # Extract user embedding tensor
            user_emb = self.user_embeddings[e]
            user_emb_exp = torch.tensor(e).expand(candidate_item_idxs.shape).to(self.device).long()

            # Fill history buffer with positive user item embeddings and
            # Remove item embeddings from candidate item set
            ignored_items = []
            for i in range(self.config.history_buffer_size):
                emb = candidate_items[pos_user_reviews[i, self.i]]
                history_buffer.push(emb.detach().clone())

            # Starting item index
            t = 0

            state = None
            action = None
            reward = None
            next_state = None
            while t < T:
                # ---------------------------- start of timestep ---------------------------- #

                # observe current state
                # choose action according to actor network or exploration
                # with torch.no_grad():
                #     state = self.state_rep_net(user_emb, torch.stack(history_buffer.to_list()))
                #     if np.random.uniform(0, 1) < self.config.eps_eval:
                #         action = torch.from_numpy(0.1 * np.random.rand(self.action_shape)).float().to(self.device)
                #     else:
                #         action = self.actor_net(state.detach())

                # Calculate ranking scores across items, discard ignored items
                ranking_scores = self.reward_function(user_emb_exp, candidate_item_idxs)
                rec_items = torch.stack(ignored_items) if len(ignored_items) > 0 else []
                ranking_scores[rec_items[:, self.i] if len(ignored_items) > 0 else []] = -float("inf")

                # Get recommended item
                rec_item_idx = torch.argmax(ranking_scores[user_reviews[:, self.i]])
                rec_item_emb = user_candidate_items[rec_item_idx]

                # Get item reward
                reward = user_reviews[rec_item_idx, self.r]

                # Track episode rewards
                rewards.append(reward.item())

                # Add item to history buffer if positive review, remove from candidate set
                # Set next state to new or old
                if reward > 0:
                    # Update history buffer with new item
                    history_buffer.push(rec_item_emb.detach().clone())
                    # Observe next state
                    # with torch.no_grad():
                    #     next_state = self.state_rep_net(user_emb, torch.stack(history_buffer.to_list()))
                # else:
                    # Keep history buffer the same, next state is current state
                    # next_state = state.detach()

                # Remove new item from future recommendations
                ignored_items.append(user_reviews[rec_item_idx])

                # Housekeeping
                t += 1
                timesteps += 1

                # ---------------------------- end of timestep ---------------------------- #

            # ---------------------------- end of episode ---------------------------- #

            # T_indicies = np.arange(T)
            # rel_real = user_reviews[T_indicies]
            # rel_real = rel_real[rel_real[:, self.r] > 0]
            rec_items = torch.stack(ignored_items)
            rel_pred = rec_items[rec_items[:, self.r] > 0]
            precision_T = len(rel_pred) / len(rec_items)

            # Logging
            epoch += 1
            e_arr.append(epoch)
            epi_precisions.append(precision_T)

            if timesteps % self.config.log_freq == 0:
                if len(rewards) > 0:
                    print(f'Episode {epoch} | '
                          f'Precision@{T} {precision_T} | '
                          f'Avg Precision@{T} {np.mean(epi_precisions):.4f} | '
                          )
                    sys.stdout.flush()

        print('Offline Evaluation Finished')
        print(f'Average Precision@{T}: {np.mean(epi_precisions):.4f} | ')
        plt.plot(e_arr, epi_precisions, label=f'Precision@{T}')
        # plt.plot(x, critic_losses, label="Test Critic")
        plt.legend()
        plt.xlabel('Episode (t)')
        plt.ylabel('Precesion@T')
        plt.title('Precision@T (Offline Evaluation)')
        plt.minorticks_on()

        # Reload model parameters
        self.load_parameters()

        return np.mean(epi_precisions)
