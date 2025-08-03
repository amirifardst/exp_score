import tensorflow as tf
from src.logging.logger import get_logger
import pandas as pd
import numpy as np
import os
from src.utils.statistics import get_statistics
make_logger = get_logger(__name__)

def calculate_exp_score(model_name,feature_maps_dict, constant,show_exp_score = True):
    make_logger.info(f"Calculating expressivity score with constant: {constant} has been started")
    # layers_name_list = feature_maps_dict.keys()
    exp_score_dict = {}
    model_exp_score_sum = 0
    

    # num_layers = len(layers_name_list)
    # make_logger.info(f"Processing {num_layers} layers for expressivity score calculation")
    num_layers_with_scores = 0
    previous_layer_expressivity_score = 0
    total_prog_score =  0 # It is used to find minimum progressivity score
    for layer_name, fmap in feature_maps_dict.items():
        try:
            make_logger.info('*'*100)
            make_logger.info(f"Processing layer {layer_name}")
            
            # Get feature map: shape (b, w, h, c)
            # Convert to TensorFlow tensor if it's not already
            if not isinstance(fmap, tf.Tensor):
                fmap = tf.convert_to_tensor(fmap, dtype=tf.float32)

            # Step1: Reshaping to (b*w*h, c) for 2D or 4D feature maps
            if fmap.ndim == 4:
                b, w, h, c = fmap.shape
                spatial_size = w * h
                # Reshape to (b*w*h, c)
                n= b * w * h
                fmap_2d = tf.reshape(fmap, (n, c))
                make_logger.info(f'fmap_2d shape for layer {layer_name}: {fmap_2d.shape}')
            elif fmap.ndim == 2:
                # If already 2D, assume shape is (n, c)
                n, c = fmap.shape
                spatial_size = 1
                fmap_2d = fmap
                make_logger.info(f'fmap_2d shape for layer {layer_name}: {fmap_2d.shape}')
            else:
                continue  # Skip if not 2D or 4D

            # Step 2: Center the features
            # Mean over rows (dim=0), keep dims for broadcasting
            mean = tf.reduce_mean(fmap_2d, axis=0, keepdims=True)
            make_logger.info(f'mean shape for layer {layer_name}: {mean.shape}')
            fmap_centered = fmap_2d - mean  # shape (n, c)
            make_logger.info(f'fmap_centered shape for layer {layer_name}: {fmap_centered.shape}')

            # Step 3: Calculate covariance matrix
            # Covariance matrix: (c, c)
            Covariance_matrix = tf.matmul(fmap_centered, fmap_centered, transpose_a=True) / tf.cast(tf.shape(fmap_centered)[0], tf.float32)
            make_logger.info(f'Covariance_matrix shape for layer {layer_name}: {Covariance_matrix.shape}')

            # Step 4: Compute eigenvalues (TensorFlow 2.4+)
            eigenvalues = tf.linalg.eigvalsh(Covariance_matrix) # shape (c,)
            eigenvalues = tf.where(eigenvalues <= 0, tf.constant(constant, dtype=tf.float32), eigenvalues)  # Ensure non-negative eigenvalues
            make_logger.info(f'eigenvalues shape for layer {layer_name}: {eigenvalues.shape}')

            # Step 5: Normalize eigenvalues to get probabilities
            prob_s = eigenvalues / tf.reduce_sum(eigenvalues)
            make_logger.info(f'prob_s shape for layer {layer_name}: {prob_s.shape}')


            # Step 6: Calculate expressivity_score using Shanon Entropy = -Seri[prob_s * log(prob_s + constant)]
            score = -prob_s * tf.math.log(prob_s + constant)
            expressivity_score = tf.reduce_sum(score).numpy().item()

            make_logger.info(f'Expressivity score for layer {layer_name}: {expressivity_score}')


            # Step 7: Normalize the expressivity score throughout the channels
            make_logger.info(f'Log (c): {tf.math.log(tf.cast(c, tf.float32))}')
            Normalized_expressivity_score = expressivity_score / tf.math.log(tf.cast(c, tf.float32)).numpy().item()
            make_logger.info(f'Normalized expressivity score for layer {layer_name}: {Normalized_expressivity_score}')   

            # Step 8: Calculate the progressivity score
    
            progressivity_score = Normalized_expressivity_score - previous_layer_expressivity_score
            previous_layer_expressivity_score = Normalized_expressivity_score

            if progressivity_score<total_prog_score:
                total_prog_score = progressivity_score  

            exp_score_dict[layer_name] = {"spatial_size": spatial_size, "num_channels": c,
                                          "log_c": tf.math.log(tf.cast(c, tf.float32)).numpy().item(),
                                          "expressivity_score":expressivity_score,
                                          "normalized_expressivity_score": Normalized_expressivity_score,
                                          "progressivity_score": progressivity_score}
            
            model_exp_score_sum += Normalized_expressivity_score

            make_logger.info(f'The score of Layer {layer_name} was added to the list')
            num_layers_with_scores += 1

        except Exception as e:
            make_logger.warning(f'N.B: The score of Layer {layer_name} was not added to the list. Exception: {e}')
            make_logger.info("*"*100)
    
    
    # Save statistics in dataframe
    exp_score_df = get_statistics(exp_score_dict, show_exp_score=show_exp_score)

    return exp_score_df


def calculate_exp_score_nas(model_names, layer_names, feature_maps_list, constant, show_exp_score=True):
    make_logger.info("=*"*100)
    make_logger.info(f"Calculating expressivity score with constant: {constant} has been started")
    make_logger.info(f"Number of architectures: {len(model_names)}")
    make_logger.info("=*"*100)
    
    all_exp_score_dict = {}
    num_architectures = len(model_names)
    for i in range(num_architectures):
        make_logger.info(f"Processing architecture {model_names[i]}")
        exp_score_dict = {}
        previous_layer_expressivity_score = 0
        counter = 0
        for fmap in feature_maps_list[i]:
            try:
                layer_name = layer_names[i][counter]
                make_logger.info(f"Processing layer_{counter+1} named: {layer_name}")

                # Get feature map: shape (b, w, h, c)
                # Convert to TensorFlow tensor if it's not already
                if not isinstance(fmap, tf.Tensor):
                    fmap = tf.convert_to_tensor(fmap, dtype=tf.float32)

                # Step1: Reshaping to (b*w*h, c) for 2D or 4D feature maps
                if fmap.ndim == 4:
                    b, c, w, h = fmap.shape
                    spatial_size = w * h
                    # Reshape to (b*w*h, c)
                    n = b * w * h
                    fmap_2d = tf.reshape(fmap, (n, c))
                    make_logger.info(f'For this layer--> [b,c,w,h]: [{b},{c},{w},{h}], spatial size: {spatial_size}, fmap_2d shape: {fmap_2d.shape}')

                elif fmap.ndim == 2:
                    # If already 2D, assume shape is (n, c)
                    n, c = fmap.shape
                    spatial_size = 1
                    fmap_2d = fmap
                    make_logger.info(f'For this layer--> [b,c,w,h]: [{b},{c},{w},{h}], spatial size: {spatial_size}, fmap_2d shape: {fmap_2d.shape}')
                else:
                    continue  # Skip if not 2D or 4D

                # Step 2: Center the features
                # Mean over rows (dim=0), keep dims for broadcasting
                mean = tf.reduce_mean(fmap_2d, axis=0, keepdims=True)
                make_logger.info(f'For this layer--> Mean.shape: {mean.shape}')
                fmap_centered = fmap_2d - mean  # shape (n, c)
                make_logger.info(f'For this layer--> fmap_centered shape: {fmap_centered.shape}')

                # Step 3: Calculate covariance matrix
                # Covariance matrix: (c, c)
                Covariance_matrix = tf.matmul(fmap_centered, fmap_centered, transpose_a=True) / tf.cast(tf.shape(fmap_centered)[0], tf.float32)
                make_logger.info(f'For this layer--> Covariance_matrix shape: {Covariance_matrix.shape}')

                # Step 4: Compute eigenvalues (TensorFlow 2.4+)
                eigenvalues = tf.linalg.eigvalsh(Covariance_matrix) # shape (c,)
                eigenvalues = tf.where(eigenvalues <= 0, tf.constant(constant, dtype=tf.float32), eigenvalues)  # Ensure non-negative eigenvalues
                make_logger.info(f'For this layer--> eigenvalues shape: {eigenvalues.shape}')

                # Step 5: Normalize eigenvalues to get probabilities
                prob_s = eigenvalues / tf.reduce_sum(eigenvalues)
                make_logger.info(f'For this layer--> prob_s shape: {prob_s.shape}')


                # Step 6: Calculate expressivity_score using Shanon Entropy = -Seri[prob_s * log(prob_s + constant)]
                score = -prob_s * tf.math.log(prob_s + constant)
                expressivity_score = tf.reduce_sum(score).numpy().item()

                make_logger.info(f'For this layer--> Expressivity score: {expressivity_score}')


                # Step 7: Normalize the expressivity score throughout the channels
                make_logger.info(f'For this layer--> Log (c): {tf.math.log(tf.cast(c, tf.float32))}')
                Normalized_expressivity_score = expressivity_score / tf.math.log(tf.cast(c, tf.float32)).numpy().item()
                make_logger.info(f'For this layer--> Normalized expressivity score: {Normalized_expressivity_score}')   

                # Step 8: Calculate the progressivity score
        
                progressivity_score = Normalized_expressivity_score - previous_layer_expressivity_score
                previous_layer_expressivity_score = Normalized_expressivity_score

                exp_score_dict[layer_name] = {"spatial_size": spatial_size, "num_channels": c,
                                            "log_c": tf.math.log(tf.cast(c, tf.float32)).numpy().item(),
                                            "expressivity_score":expressivity_score,
                                            "normalized_expressivity_score": Normalized_expressivity_score,
                                            "progressivity_score": progressivity_score}
                

                make_logger.info(f'The score of {layer_name} was added to the list')
                counter += 1                    
            except Exception as e:
                make_logger.warning(f'N.B: The score of {layer_name} was not added to the list. Exception: {e}')
                make_logger.info("*"*100)


        make_logger.info(f"Expressivity scores was calculated for architecture {model_names[i]}")
        make_logger.info('-'*100)  
        # Save statistics in dataframe
        exp_score_df = get_statistics(exp_score_dict, show_exp_score=show_exp_score)

        all_exp_score_dict[model_names[i]] = exp_score_df

    make_logger.info(f"Expressivity scores for all architectures were calculated successfully")
    make_logger.info("=*"*100)
    return all_exp_score_dict