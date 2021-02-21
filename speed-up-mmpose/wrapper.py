from collections import namedtuple
from copy import deepcopy
import os
import argparse

from tbad.gpu import configure_gpu_resources
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from functools import reduce

from tbad.autoencoder.data import load_trajectories, extract_global_features, change_coordinate_system
from tbad.autoencoder.data import scale_trajectories, load_anomaly_masks, assemble_ground_truth_and_reconstructions
from tbad.autoencoder.data import quantile_transform_errors
from tbad.autoencoder.data import  Trajectory, assemble_result
from tbad.rnn_autoencoder.data import remove_short_trajectories, aggregate_rnn_ae_evaluation_data
from tbad.rnn_autoencoder.data import compute_rnn_ae_reconstruction_errors, summarise_reconstruction_errors
from tbad.rnn_autoencoder.data import summarise_reconstruction, retrieve_future_skeletons
from tbad.rnn_autoencoder.data import discard_information_from_padded_frames
from tbad.combined_model.fusion import load_pretrained_combined_model
from tbad.combined_model.data import inverse_scale, restore_global_coordinate_system, restore_original_trajectory
from tbad.combined_model.data import write_reconstructed_trajectories, detect_most_anomalous_or_most_normal_frames
from tbad.combined_model.data import compute_num_frames_per_video, write_predicted_masks, compute_worst_mistakes
from tbad.combined_model.data import write_worst_mistakes, clip_trajectories, normalise_errors_by_bounding_box_area
from tbad.visualisation import compute_bounding_box


V_01 = [1] * 75 + [0] * 46 + [1] * 269 + [0] * 47 + [1] * 427 + [0] * 47 + [1] * 20 + [0] * 70 + [1] * 438  # 1439 Frames
V_02 = [1] * 272 + [0] * 48 + [1] * 403 + [0] * 41 + [1] * 447  # 1211 Frames
V_03 = [1] * 293 + [0] * 48 + [1] * 582  # 923 Frames
V_04 = [1] * 947  # 947 Frames
V_05 = [1] * 1007  # 1007 Frames
V_06 = [1] * 561 + [0] * 64 + [1] * 189 + [0] * 193 + [1] * 276  # 1283 Frames
V_07_to_15 = [1] * 6457
V_16 = [1] * 728 + [0] * 12  # 740 Frames
V_17_to_21 = [1] * 1317
AVENUE_MASK = np.array(V_01 + V_02 + V_03 + V_04 + V_05 + V_06 + V_07_to_15 + V_16 + V_17_to_21) == 1

class AnomalyDetector():
  def __init__(self,
               model_path,
               resolution):
    arg_parser = self.create_arg_parser()
    self.args = arg_parser.parse_args(["combined_model",
                                       model_path,
                                       "--video_resolution",
                                       resolution,
                                       "--overlapping_trajectories"
                                      ])
    configure_gpu_resources(self.args.gpu_ids, self.args.gpu_memory_fraction)
    
    model_info = os.path.basename(os.path.split(model_path)[0])
    message_passing = 'mp' in model_info

    self.pretrained_combined_model, self.global_scaler, self.local_scaler, self.out_scaler = \
        load_pretrained_combined_model(model_path, message_passing=message_passing)

    video_resolution = [float(measurement) for measurement in self.args.video_resolution.split('x')]
    self.video_resolution = np.array(video_resolution, dtype=np.float32)
    self.overlapping_trajectories = self.args.overlapping_trajectories

    # Extract information about the models
    self.reconstruct_original_data = 'down' in model_info
    self.global_normalisation_strategy = 'zero_one'
    if '_G3stds_' in model_info:
        self.global_normalisation_strategy = 'three_stds'
    elif '_Grobust_' in model_info:
        self.global_normalisation_strategy = 'robust'

    self.local_normalisation_strategy = 'zero_one'
    if '_L3stds_' in model_info:
        self.local_normalisation_strategy = 'three_stds'
    elif '_Lrobust_' in model_info:
        self.local_normalisation_strategy = 'robust'

    self.out_normalisation_strategy = 'zero_one'
    if '_O3stds_' in model_info:
        self.out_normalisation_strategy = 'three_stds'
    elif '_Orobust_' in model_info:
        self.out_normalisation_strategy = 'robust'

    self.multiple_outputs = self.pretrained_combined_model.multiple_outputs
    self.input_length, self.rec_length = self.pretrained_combined_model.input_length, self.pretrained_combined_model.reconstruction_length
    self.input_gap, self.pred_length = 0, self.pretrained_combined_model.prediction_length
    self.reconstruct_reverse = self.pretrained_combined_model.reconstruct_reverse
    self.loss = self.pretrained_combined_model.loss

  def create_arg_parser(self):
    parser = argparse.ArgumentParser(description='Functions for Evaluation of Trained Trajectory-Based Anomaly Models.')

    gp_gpu = parser.add_argument_group('GPU')
    gp_gpu.add_argument('--gpu_ids', default='0', type=str, help='Which GPUs to use.')
    gp_gpu.add_argument('--gpu_memory_fraction', default=0.20, type=float,
                        help='Fraction of the memory to grab from each GPU.')

    subparsers = parser.add_subparsers(title='sub-commands', description='Valid sub-commands.')

    # Create a sub-parser for evaluation of a trained Combined model
    parser_combined_model = subparsers.add_parser('combined_model',
                                                  help='Evaluate a Trained Combined Model.')
    parser_combined_model.add_argument('pretrained_model', type=str,
                                       help='Directory containing pre-trained model architecture definition, model '
                                            'weights and data scaler.')
    parser_combined_model.add_argument('--video_resolution', default='856x480', type=str,
                                       help='Resolution of the trajectories\' original video(s). It should be '
                                            'specified as WxH, where W is the width and H the height of the video.')
    parser_combined_model.add_argument('--overlapping_trajectories', action='store_true')

    gp_combined_model_logging = parser_combined_model.add_argument_group('Evaluation Logging')
    gp_combined_model_logging.add_argument('--write_reconstructions', action='store_true')
    gp_combined_model_logging.add_argument('--write_bounding_boxes', action='store_true')
    gp_combined_model_logging.add_argument('--write_predictions', action='store_true')
    gp_combined_model_logging.add_argument('--write_predictions_bounding_boxes', action='store_true')
    gp_combined_model_logging.add_argument('--write_anomaly_masks', action='store_true')
    gp_combined_model_logging.add_argument('--write_mistakes', action='store_true')

    # Create a sub-parser for evaluation of multiple pre-trained Combined models
    parser_combined_models = subparsers.add_parser('combined_models',
                                                   help='Evaluate Multiple Pre-Trained Combined Models.')
    parser_combined_models.add_argument('pretrained_models', type=str,
                                        help='Directory containing a folder for each pre-trained model.')
    parser_combined_models.add_argument('--video_resolution', default='856x480', type=str,
                                        help='Resolution of the trajectories\' original video(s). It should be '
                                             'specified as WxH, where W is the width and H the height of the video.')
    parser_combined_models.add_argument('--overlapping_trajectories', action='store_true')

    gp_combined_models_logging = parser_combined_models.add_argument_group('Evaluation Logging')
    gp_combined_models_logging.add_argument('--write_reconstructions', action='store_true')
    gp_combined_models_logging.add_argument('--write_bounding_boxes', action='store_true')
    gp_combined_models_logging.add_argument('--write_predictions', action='store_true')
    gp_combined_models_logging.add_argument('--write_predictions_bounding_boxes', action='store_true')
    gp_combined_models_logging.add_argument('--write_anomaly_masks', action='store_true')
    gp_combined_models_logging.add_argument('--write_mistakes', action='store_true')

    return parser

  def eval_combined_model(self, fl, skeletons):
      trajectories = {}
      for traj_id, traj_con in skeletons.items():
        trajectories[traj_id] = Trajectory(traj_id, traj_con["frames"], np.array(traj_con["coordinates"]))

      trajectories = remove_short_trajectories(trajectories, input_length=self.input_length,
                                            input_gap=self.input_gap, pred_length=self.pred_length)

      global_trajectories = extract_global_features(deepcopy(trajectories), video_resolution=self.video_resolution)
      global_trajectories = change_coordinate_system(global_trajectories, video_resolution=self.video_resolution,
                                                    coordinate_system='global', invert=False)

      trajectories_ids, frames, X_global = \
          aggregate_rnn_ae_evaluation_data(global_trajectories,
                                          input_length=self.input_length,
                                          input_gap=self.input_gap,
                                          pred_length=self.pred_length,
                                          overlapping_trajectories=self.overlapping_trajectories)

      X_global, _ = scale_trajectories(X_global, scaler=self.global_scaler, strategy=self.global_normalisation_strategy)

      local_trajectories = deepcopy(trajectories)
      local_trajectories = change_coordinate_system(local_trajectories, video_resolution=self.video_resolution,
                                                    coordinate_system='bounding_box_centre', invert=False)
      _, _, X_local = aggregate_rnn_ae_evaluation_data(local_trajectories, input_length=self.input_length,
                                                      input_gap=self.input_gap, pred_length=self.pred_length,
                                                      overlapping_trajectories=self.overlapping_trajectories)
      X_local, _ = scale_trajectories(X_local, scaler=self.local_scaler, strategy=self.local_normalisation_strategy)

      original_trajectories = deepcopy(trajectories)
      _, _, X_original = aggregate_rnn_ae_evaluation_data(original_trajectories, input_length=self.input_length,
                                                          input_gap=self.input_gap, pred_length=self.pred_length,
                                                          overlapping_trajectories=self.overlapping_trajectories)

      if self.reconstruct_original_data:
          out_trajectories = trajectories
          out_trajectories = change_coordinate_system(out_trajectories, video_resolution=self.video_resolution,
                                                      coordinate_system='global', invert=False)
          _, _, X_out = aggregate_rnn_ae_evaluation_data(out_trajectories, input_length=self.input_length,
                                                        input_gap=self.input_gap, pred_length=self.pred_length,
                                                        overlapping_trajectories=self.overlapping_trajectories)
          X_out, _ = scale_trajectories(X_out, scaler=self.out_scaler, strategy=self.out_normalisation_strategy)

      # Reconstruct
      X_input = [X_global, X_local]

      if self.pred_length == 0:
          if self.multiple_outputs:
              _, _, reconstructed_X = self.pretrained_combined_model.predict(X_input, batch_size=1024)
          else:
              reconstructed_X = self.pretrained_combined_model.predict(X_input, batch_size=1024)
      else:
          if self.multiple_outputs:
              _, _, reconstructed_X, _, _, predicted_y = \
                  self.pretrained_combined_model.predict(X_input, batch_size=1024)
          else:
              reconstructed_X, predicted_y = self.pretrained_combined_model.predict(X_input, batch_size=1024)

      if self.reconstruct_reverse:
          reconstructed_X = reconstructed_X[:, ::-1, :]

      X = X_out if self.reconstruct_original_data else np.concatenate((X_global, X_local), axis=-1)
      reconstruction_errors = compute_rnn_ae_reconstruction_errors(X[:, :self.rec_length, :], reconstructed_X, self.loss)
      reconstruction_ids, reconstruction_frames, reconstruction_errors = \
          summarise_reconstruction_errors(reconstruction_errors, frames[:, :self.rec_length], trajectories_ids[:, :self.rec_length])

      # reconstruction_errors = reconstruction_errors.reshape((len(skeletons), -1))
      # reconstruction_errors = reduce(lambda x, y: np.maximum(x, y), reconstruction_errors[1:], reconstruction_errors[0])
      y_hat = assemble_result(fl, reconstruction_ids, reconstruction_frames, reconstruction_errors)

      # print('Reconstruction Based:')
      # print(y_hat)

      if self.pred_length > 0:
          predicted_frames = frames[:, :self.pred_length] + self.input_length
          predicted_ids = trajectories_ids[:, :self.pred_length]
          
          y = retrieve_future_skeletons(trajectories_ids, X, self.pred_length)
          pred_errors = compute_rnn_ae_reconstruction_errors(y, predicted_y, self.loss)

          pred_ids, pred_frames, pred_errors = discard_information_from_padded_frames(predicted_ids,
                                                                                      predicted_frames,
                                                                                      pred_errors, self.pred_length)

          pred_ids, pred_frames, pred_errors = summarise_reconstruction_errors(pred_errors, pred_frames, pred_ids)

          # pred_errors = pred_errors.reshape((len(skeletons), -1))
          # pred_errors = reduce(lambda x, y: np.maximum(x, y), pred_errors[1:], pred_errors[0])
          # pred_errors = np.pad(pred_errors, (reconstruction_errors.shape[0] - pred_errors.shape[0], 0), 'constant', constant_values=0)

          y_hat_pred = assemble_result(fl, pred_ids, pred_frames, pred_errors)
          # print('Prediction Based:')
          # print(y_hat_pred)

          # comb_errors = reconstruction_errors + pred_errors
          y_hat_comb = y_hat + y_hat_pred
          # print('Reconstruction + Prediction Based:')
          # print(y_hat_comb)

      if self.pred_length > 0:
          return y_hat_comb
      else:
          return y_hat
