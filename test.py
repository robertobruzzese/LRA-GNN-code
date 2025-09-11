import torch
from torch.utils.data import DataLoader
from models.lra_gnn import LRA_GNN
from models.progressive_rl import ProgressiveRLAgent
from dataset import AgeEstimationDataset
from utils import calculate_metrics
import matplotlib.pyplot as plt
import os

def test_model(model, rl_agent, test_loader, device):
    
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass through LRA-GNN
            outputs = model(images)
            
            # Collect predictions and labels
            all_predictions.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    # Concatenate all predictions and labels
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # Calculate metrics
    metrics = calculate_metrics(torch.tensor(all_predictions), torch.tensor(all_labels))
    print("Test Metrics:")
    print(f"MAE: {metrics['MAE']:.2f}")
    print(f"CS_5: {metrics['CS_5']:.2f}%")
    print(f"Epsilon_Error: {metrics['Epsilon_Error']:.4f}")
    
    # Visualize results
    plt.figure(figsize=(10, 6))
    plt.scatter(all_labels, all_predictions, alpha=0.5)
    plt.plot([min(all_labels), max(all_labels)], [min(all_labels), max(all_labels)], color='red', linestyle='--')
    plt.xlabel('Actual Age')
    plt.ylabel('Predicted Age')
    plt.title('Actual vs. Predicted Age')
    plt.show()


if __name__ == "__main__":
    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define data transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Initialize the dataset
    dataset = AgeEstimationDataset(root_dir='path/to/dataset', transform=transform)
    
    # Create data loader
    batch_size = 32
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize the LRA-GNN model
    model = LRA_GNN(num_layers=12, num_heads=8, in_channels=3, hidden_channels=128, out_channels=1)
    model.to(device)
    
    # Load the trained model
    model.load_state_dict(torch.load('path/to/save/model.pth'))
    
    # Initialize the Progressive RL agent
    state_dim = 128  # Example state dimension (output from LRA-GNN)
    action_dim = 5   # Example action dimension (age groups)
    rl_agent = ProgressiveRLAgent(state_dim, action_dim)
    
    # Test the model
    test_model(model, rl_agent, test_loader, device)