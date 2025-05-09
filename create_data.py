import torch
from torchvision import datasets, transforms
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

################################################# GENERATE DATASETS #################################################
'''
    This script is used to generate different types of datasets from the FashionMNIST dataset.
    We will be generating 3 different types of datasets:
    1. Balanced Dataset
    2. Semi-Imbalanced Dataset
        - This will randomly select two classes and make them minority 
        - We can also spcificy the exact classes if we want to 
        - I am setting the majority 8 to have 600 each
        - The two minority will have 100 each
        - Class 2 and 6 are the minority 


    3. Highly Imbalanced Dataset
        Class distribution:
            Class 0: 870 samples (17.27%)
            Class 1: 677 samples (13.44%)
            Class 2: 161 samples (3.20%)
            Class 3: 628 samples (12.46%)
            Class 4: 319 samples (6.33%)
            Class 5: 937 samples (18.59%)
            Class 6: 775 samples (15.38%)
            Class 7: 54 samples (1.07%)
            Class 8: 499 samples (9.90%)
            Class 9: 119 samples (2.36%)

            Train Set Distribution
                Total samples: 4028

                Class distribution:
                Class 0: 696 samples (17.28%)
                Class 1: 541 samples (13.43%)
                Class 2: 128 samples (3.18%)
                Class 3: 502 samples (12.46%)
                Class 4: 255 samples (6.33%)
                Class 5: 749 samples (18.59%)
                Class 6: 620 samples (15.39%)
                Class 7: 43 samples (1.07%)
                Class 8: 399 samples (9.91%)
                Class 9: 95 samples (2.36%)

                Analyzing test set distribution:

            Test Set Distribution
                Total samples: 1011

                Class distribution:
                Class 0: 174 samples (17.21%)
                Class 1: 136 samples (13.45%)
                Class 2: 33 samples (3.26%)
                Class 3: 126 samples (12.46%)
                Class 4: 64 samples (6.33%)
                Class 5: 188 samples (18.60%)
                Class 6: 155 samples (15.33%)
                Class 7: 11 samples (1.09%)
                Class 8: 100 samples (9.89%)
                Class 9: 24 samples (2.37%)

    Then for each of the data sets we create a training and testing data set. They will have the same distribution.
    

    The FashionMNIST has the labels 0-9
    0 -- T-Shirt
    1 -- Trouser
    2 -- Pullover 
    3 -- Dress 
    4 -- Coat
    5 -- Sandal 
    6 -- Shirt 
    7 -- Sneaker 
    8 -- Bag 
    9 -- Ankle Boot

    Each class has 6000 examples.

'''
#####################################################################################################################

def analyze_dataset(dataset, title="Dataset Distribution"):
    """Analyze and plot the distribution of classes in a dataset."""
    labels = [label for _, label in dataset]
    class_counts = Counter(labels)
    
    # Print statistics
    print(f"\n{title}")
    print("Total samples:", len(dataset))
    print("\nClass distribution:")
    for class_idx, count in sorted(class_counts.items()):
        print(f"Class {class_idx}: {count} samples ({count/len(dataset)*100:.2f}%)")
    
    # Plot distribution
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(class_counts.keys()), y=list(class_counts.values()))
    plt.title(title)
    plt.xlabel("Class")
    plt.ylabel("Number of Samples")
    plt.show()
    
    return class_counts

def create_balanced_dataset(dataset, samples_per_class=500):
    """Create a balanced dataset with equal samples per class."""
    class_indices = {i: [] for i in range(10)}
    
    # Collect indices for each class
    for idx, (_, label) in enumerate(dataset):
        # this is adding the index of where all the labels are in the data set 
        class_indices[label].append(idx)
    
    # Select equal number of samples from each class
    selected_indices = []
    for class_idx in range(10):
        selected_indices.extend(np.random.choice(class_indices[class_idx], 
                                               size=min(samples_per_class, len(class_indices[class_idx])), 
                                               replace=False))
    
    return torch.utils.data.Subset(dataset, selected_indices)

def create_semi_imbalanced_dataset(dataset, majority_samples=500, minority_samples=100, minority_classes=None):
    """Create a semi-imbalanced dataset where most classes are balanced but 1-2 classes are underrepresented.
    
    Args:
        dataset: The full dataset
        majority_samples: Number of samples for majority classes (default: 500)
        minority_samples: Number of samples for minority classes (default: 100)
        minority_classes: List of class indices to be underrepresented (default: None, will randomly select 2 classes)
    """
    class_indices = {i: [] for i in range(10)}
    
    # Collect indices for each class
    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)
    
    # If minority_classes not specified, randomly select 2 classes
    if minority_classes is None:
        minority_classes = np.random.choice(10, size=2, replace=False)
    
    selected_indices = []
    
    # Handle each class
    for class_idx in range(10):
        if class_idx in minority_classes:
            # For minority classes, use minority_samples
            samples = minority_samples
        else:
            # For majority classes, use majority_samples
            samples = majority_samples
            
        if len(class_indices[class_idx]) >= samples:
            selected_indices.extend(np.random.choice(class_indices[class_idx], 
                                                   size=samples, 
                                                   replace=False))
    
    return torch.utils.data.Subset(dataset, selected_indices)

def create_highly_imbalanced_dataset(dataset, min_samples=50, max_samples=1000):
    """Create a highly imbalanced dataset with random class sizes."""
    class_indices = {i: [] for i in range(10)}
    
    # Collect indices for each class
    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)
    
    # Randomly assign number of samples for each class
    selected_indices = []
    for class_idx in range(10):
        samples = np.random.randint(min_samples, max_samples + 1)
        if len(class_indices[class_idx]) >= samples:
            selected_indices.extend(np.random.choice(class_indices[class_idx], 
                                                   size=samples, 
                                                   replace=False))
    
    return torch.utils.data.Subset(dataset, selected_indices)

def split_dataset(dataset, train_ratio=0.8):
    """Split dataset into train and test sets while maintaining class distribution."""
    # Get all indices and labels
    indices = list(range(len(dataset)))
    labels = [label for _, label in dataset]
    
    # Create train and test indices for each class
    train_indices = []
    test_indices = []
    
    # Group indices by class
    class_indices = {i: [] for i in range(10)}
    for idx, label in enumerate(labels):
        class_indices[label].append(idx)
    
    # Split each class's indices
    for class_idx in range(10):
        class_indices_list = class_indices[class_idx]
        np.random.shuffle(class_indices_list)
        
        # Calculate split point
        split_point = int(len(class_indices_list) * train_ratio)
        
        # Add to train and test sets
        train_indices.extend(class_indices_list[:split_point])
        test_indices.extend(class_indices_list[split_point:])
    
    # Create train and test subsets
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    
    return train_dataset, test_dataset

def main():
    # Load FashionMNIST dataset
    torch.serialization.add_safe_globals(["Subset"])
    # transform = transforms.ToTensor()
    # full_dataset = datasets.FashionMNIST(root='./data', train=True, 
                                       # download=True, transform=transform)
    
    # Analyze original dataset
    # print("Analyzing original FashionMNIST dataset...")
    # original_distribution = analyze_dataset(full_dataset, "Original FashionMNIST Distribution")
    
    # Create and analyze balanced dataset
    # print("\nCreating balanced dataset...")
    # balanced_dataset = create_balanced_dataset(full_dataset, samples_per_class=500)
    
    # Create and analyze semi-imbalanced dataset
    # print("\nCreating semi-imbalanced dataset...")
    # You can specify which classes should be minority classes, or let it randomly select
    # minority_classes = [2, 7]  # Example: Pullover and Sneaker will be underrepresented
    # semi_imbalanced_dataset = create_semi_imbalanced_dataset(full_dataset, 
                                                         #   majority_samples=600,
                                                         #   minority_samples=100)
    # torch.save(semi_imbalanced_dataset, 'expirement_data/semi_imbalanced_dataset.pt')
    # print("Testing")
    
    # test = torch.load('expirement_data/semi_imbalanced_dataset.pt', weights_only=False)
    # analyze_dataset(test, "Semi-Imbalanced Dataset Distribution")
    # Create and analyze highly imbalanced dataset
    # print("\nCreating highly imbalanced dataset...")
    # highly_imbalanced_dataset = create_highly_imbalanced_dataset(full_dataset, 
                                                                  # min_samples=50, 
                                                                  # max_samples=1000)
    
    
    # torch.save(highly_imbalanced_dataset, 'expirement_data/highly_imbalanced_dataset.pt')
    # test = torch.load('expirement_data/highly_imbalanced_dataset.pt', weights_only=False)
    # analyze_dataset(test, "Highly Imbalanced Dataset Distribution")
    # Save the datasets
    # torch.save(balanced_dataset, 'expirement_data/balanced_dataset.pt')
    # torch.save(semi_imbalanced_dataset, 'expirement_data/semi_imbalanced_dataset.pt')
    # torch.save(highly_imbalanced_dataset, 'expirement_data/highly_imbalanced_dataset.pt')
    
    # print("\nDatasets have been saved as:")
    # print("- expirement_data/balanced_dataset.pt")
    # print("- expirement_data/semi_imbalanced_dataset.pt")
    # print("- expirement_data/highly_imbalanced_dataset.pt")

    # now we load in the data and create a train and test 
    balanced = torch.load('expirement_data/balanced/balanced_dataset.pt', weights_only=False)
    
    # Split the balanced dataset into train and test sets
    train_dataset, test_dataset = split_dataset(balanced, train_ratio=0.8)
    
    # Analyze the splits
    print("\nAnalyzing train set distribution:")
    analyze_dataset(train_dataset, "Train Set Distribution")
    
    print("\nAnalyzing test set distribution:")
    analyze_dataset(test_dataset, "Test Set Distribution")
    
    # Save the splits
    torch.save(train_dataset, 'expirement_data/balanced/train_dataset.pt')
    torch.save(test_dataset, 'expirement_data/balanced/test_dataset.pt')
    
    print("\nTrain and test sets have been saved as:")
    print("- expirement_data/high/train_dataset.pt")
    print("- expirement_data/high/test_dataset.pt")

if __name__ == "__main__":
    main()
