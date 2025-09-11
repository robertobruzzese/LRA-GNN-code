import torch
import numpy as np

def calculate_metrics(predictions, labels):
  
    # Convert predictions and labels to numpy arrays
    predictions = predictions.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    
    # Calculate Mean Absolute Error (MAE)
    mae = np.mean(np.abs(predictions - labels))
    
    # Calculate Cumulative Score (CS) for a threshold of 5 years
    cs_5 = np.mean(np.abs(predictions - labels) <= 5) * 100
    
    # Calculate Normal Score (𝜖-error) for datasets with mean and variance
    epsilon_error = np.mean(np.exp(-0.5 * ((predictions - labels) ** 2)))
    
    metrics = {
        'MAE': mae,
        'CS_5': cs_5,
        'Epsilon_Error': epsilon_error
    }
    
    return metrics

def preprocess_keypoints(keypoints, image_size=(224, 224)):
   
    # Normalize keypoints to the range [0, 1]
    keypoints = keypoints.float()
    keypoints[:, 0] /= image_size[0]
    keypoints[:, 1] /= image_size[1]
    
    return keypoints


if __name__ == "__main__":
    # Example predictions and labels
    predictions = torch.tensor([25.0, 30.0, 35.0, 40.0])
    labels = torch.tensor([24.0, 31.0, 34.0, 41.0])
    
    # Calculate metrics
    metrics = calculate_metrics(predictions, labels)
    print("Evaluation Metrics:")
    print(metrics)
    
    # Example keypoints
    keypoints = torch.tensor([
        [50, 50],
        [100, 50],
        [75, 100],
        [125, 100]
    ])
    
    # Preprocess keypoints
    preprocessed_keypoints = preprocess_keypoints(keypoints)
    print("Preprocessed Keypoints:")
    print(preprocessed_keypoints)