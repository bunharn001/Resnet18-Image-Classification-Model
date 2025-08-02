import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
import json

# Loading json attribute
with open('config.json', 'r') as file:
    data = json.load(file)

# Load Paths
data_dir = data['data_dir']
model_path = data['saved_model_path']
classes = data['classes']

# Predict directory
predict_dir = os.path.join(data_dir, "predict")
result_dir = os.path.join(data_dir, "result")
metrics_dir = os.path.join(data_dir, "metrics")

# Create directories if they don't exist
os.makedirs(result_dir, exist_ok=True)
os.makedirs(metrics_dir, exist_ok=True)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Model setup
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# Transform (should match training preprocessing)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
])

# Font for label
try:
    font = ImageFont.truetype("arial.ttf", 36)  # Smaller font size
except:
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 36)
    except:
        font = ImageFont.load_default()

# Storage for predictions and ground truth
all_predictions = []
all_probabilities = []
all_true_labels = []
all_filenames = []
all_confidences = []
counters = {cls: 1 for cls in classes}

print("Starting inference...")

# Inference loop
for fname in os.listdir(predict_dir):
    fpath = os.path.join(predict_dir, fname)
    if os.path.isfile(fpath) and fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
        
        # Extract true label from filename (format: class_name_number.extension)
        true_label = None
        filename_without_ext = os.path.splitext(fname)[0]  # Remove file extension
        
        # Method 1: Try exact class name match at start of filename
        for i, cls in enumerate(classes):
            if filename_without_ext.lower().startswith(cls.lower() + '_'):
                true_label = i
                break
        
        # Method 2: If no exact match, try partial match (fallback)
        if true_label is None:
            for i, cls in enumerate(classes):
                if cls.lower() in filename_without_ext.lower():
                    true_label = i
                    break
        
        # Load and preprocess image
        img = Image.open(fpath).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(img_tensor)
            probabilities = torch.softmax(output, dim=1)
            _, predicted = torch.max(output, 1)
            
            class_idx = predicted.item()
            class_name = classes[class_idx]
            confidence = probabilities[0][class_idx].item()
            
            # Store predictions
            all_predictions.append(class_idx)
            all_probabilities.append(probabilities[0].cpu().numpy())
            all_filenames.append(fname)
            all_confidences.append(confidence)
            
            if true_label is not None:
                all_true_labels.append(true_label)

        # Draw label with confidence on image (like old version)
        img_draw = img.copy()
        draw = ImageDraw.Draw(img_draw)
        
        # Create label text
        label_text = f"{class_name}"
        confidence_text = f"Confidence: {confidence:.1%}"
        
        # Get image dimensions
        img_width, img_height = img_draw.size
        
        # Draw label at top
        label_bbox = draw.textbbox((0, 0), label_text, font=font)
        label_width = label_bbox[2] - label_bbox[0]
        label_height = label_bbox[3] - label_bbox[1]
        
        # Position label at top center
        label_x = (img_width - label_width) // 2
        label_y = 10
        
        # Draw background rectangle for label
        margin = 8
        draw.rectangle(
            [label_x - margin, label_y - margin, 
             label_x + label_width + margin, label_y + label_height + margin],
            fill=(0, 0, 0, 180)
        )
        draw.text((label_x, label_y), label_text, fill=(255, 255, 255), font=font)
        
        # Draw confidence at bottom
        conf_bbox = draw.textbbox((0, 0), confidence_text, font=font)
        conf_width = conf_bbox[2] - conf_bbox[0]
        conf_height = conf_bbox[3] - conf_bbox[1]
        
        conf_x = (img_width - conf_width) // 2
        conf_y = img_height - conf_height - 10
        
        # Draw background rectangle for confidence
        draw.rectangle(
            [conf_x - margin, conf_y - margin,
             conf_x + conf_width + margin, conf_y + conf_height + margin],
            fill=(0, 0, 0, 180)
        )
        draw.text((conf_x, conf_y), confidence_text, fill=(255, 255, 255), font=font)

        # Save with original filename but in result directory
        result_path = os.path.join(result_dir, fname)
        img_draw.save(result_path)

        # Delete original image
        os.remove(fpath)

        print(f"Processed: {fname} → {class_name} (confidence: {confidence:.1%})")

print(f"\nInference complete! Processed {len(all_predictions)} images.")

# Create CSV file with confidence data
df_results = pd.DataFrame({
    'Filename': all_filenames,
    'Predicted_Class': [classes[pred] for pred in all_predictions],
    'Actual_Class': [classes[true_label] if true_label < len(all_true_labels) and true_label < len(classes) else 'Unknown' 
                     for true_label in (all_true_labels if len(all_true_labels) == len(all_predictions) else ['Unknown'] * len(all_predictions))],
    'Confidence': [f"{conf:.1%}" for conf in all_confidences],
    'Confidence_Score': all_confidences
})

# Add individual class probabilities
for i, class_name in enumerate(classes):
    df_results[f'{class_name}_Probability'] = [prob[i] for prob in all_probabilities]

# Save to CSV
csv_path = os.path.join(metrics_dir, 'predictions_confidence.csv')
df_results.to_csv(csv_path, index=False)
print(f"✅ Confidence data saved to CSV: {csv_path}")

# Calculate and display metrics only if we have ground truth labels
if len(all_true_labels) == len(all_predictions) and len(all_true_labels) > 0:
    print("\n" + "="*60)
    print("EVALUATION METRICS")
    print("="*60)
    
    # Convert to numpy arrays
    y_true = np.array(all_true_labels)
    y_pred = np.array(all_predictions)
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    
    print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Macro Precision: {macro_precision:.4f}")
    print(f"Macro Recall: {macro_recall:.4f}")
    print(f"Macro F1-Score: {macro_f1:.4f}")
    
    print("\nPer-Class Metrics:")
    print("-" * 50)
    for i, class_name in enumerate(classes):
        print(f"{class_name:>10}: Precision={precision[i]:.4f}, Recall={recall[i]:.4f}, F1={f1[i]:.4f}, Support={support[i]}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"\nConfusion Matrix:")
    print("-" * 30)
    print("Predicted →")
    print("Actual ↓")
    
    # Print confusion matrix with class names
    header = "    " + "".join([f"{cls:>8}" for cls in classes])
    print(header)
    for i, class_name in enumerate(classes):
        row = f"{class_name:>4}" + "".join([f"{cm[i][j]:>8}" for j in range(len(classes))])
        print(row)
    
    # Calculate per-class accuracy from confusion matrix
    print(f"\nPer-Class Accuracy:")
    print("-" * 25)
    for i, class_name in enumerate(classes):
        class_acc = cm[i][i] / np.sum(cm[i]) if np.sum(cm[i]) > 0 else 0
        print(f"{class_name}: {class_acc:.4f} ({class_acc*100:.2f}%)")
    
    # Detailed classification report
    report_text = classification_report(y_true, y_pred, target_names=classes)
    print(f"\nDetailed Classification Report:")
    print("-" * 40)
    print(report_text)
    
    # Create and save confusion matrix plot with matplotlib
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Class', fontsize=12)
    plt.ylabel('True Class', fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save confusion matrix plot
    cm_path = os.path.join(metrics_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Confusion matrix saved: {cm_path}")
    
    # Create normalized confusion matrix
    plt.figure(figsize=(10, 8))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=classes, yticklabels=classes,
                cbar_kws={'label': 'Normalized Count'})
    plt.title('Normalized Confusion Matrix', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Class', fontsize=12)
    plt.ylabel('True Class', fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save normalized confusion matrix
    cm_norm_path = os.path.join(metrics_dir, 'confusion_matrix_normalized.png')
    plt.savefig(cm_norm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Normalized confusion matrix saved: {cm_norm_path}")
    
else:
    print("\n⚠️  Ground truth labels not available or incomplete.")
    print("Creating prediction distribution visualization instead...")
    
    # Create a simple prediction distribution plot when no ground truth is available
    unique, counts = np.unique(all_predictions, return_counts=True)
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar([classes[i] for i in unique], counts, color='skyblue', alpha=0.7)
    plt.title('Prediction Distribution', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Classes', fontsize=12)
    plt.ylabel('Number of Images', fontsize=12)
    plt.xticks(rotation=45)
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                str(count), ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    pred_dist_path = os.path.join(metrics_dir, 'prediction_distribution.png')
    plt.savefig(pred_dist_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Prediction distribution saved: {pred_dist_path}")

# Print prediction distribution
print(f"\nPrediction Distribution:")
print("-" * 25)
unique, counts = np.unique(all_predictions, return_counts=True)
for class_idx, count in zip(unique, counts):
    percentage = (count / len(all_predictions)) * 100
    print(f"{classes[class_idx]}: {count} images ({percentage:.1f}%)")

print(f"\n✅ All results saved!")
print(f"  - Labeled images: {result_dir}")
print(f"  - Confusion matrix plots: {metrics_dir}")
print(f"  - Confidence data (CSV): predictions_confidence.csv")