from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch._dynamo
torch._dynamo.config.suppress_errors = True


class Detector:
    def __init__(self, tokenizer_name, detector_path, device):
        """
        Initializes the Detector with a pre-trained model and tokenizer.
        
        Args:
            tokenizer_name (str): Tokenizer name.
            detector_path (str): Pre-trained model name or path.
            device (str): Device for computation, e.g., "cuda:0" or "cpu".
        """
        self.device = device

        # Load pre-trained model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(detector_path)
        self.model.to(self.device)
    
    def predict_batch(self, batch, cls_text_key, max_length, threshold=0.5, temperature=None):
        """
        Runs the model on a batch of data to make predictions.

        Args:
            batch (dict): A dictionary containing the text data to classify.
            cls_text_key (str): The key in the batch dictionary where the input text is stored.
            max_length (int): Maximum sequence length for tokenization.
        
        Returns:
            dict: Contains 'cls_score' and 'cls_confidence' for each input text.
        """
        self.tokenizer.truncation_side='left'
        # Tokenize all 'cls_text' in the batch
        inputs = self.tokenizer(batch[cls_text_key], padding=True, truncation=True, return_tensors="pt", max_length=max_length)
        
        # Move inputs to the same device as the model (GPU/CPU)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        
        # Run the model (ensure no gradient tracking)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Get logits and convert them to softmax to get confidence
        logits = outputs.logits

        if temperature is not None:
            logits = logits / temperature

        cls_confidences = torch.softmax(logits, dim=-1)[: , 1]  # Confidence scores
        cls_scores = (cls_confidences > threshold).int().tolist()

        # Return the results
        return {
            'cls_score': cls_scores,
            'cls_confidence': cls_confidences.tolist()
        }
    
    def evaluate(self, classified_dataset, data_label):
        """
        Evaluates the model's performance on a dataset.

        Args:
            classified_dataset (Dataset): The dataset containing predictions and true labels.
        
        Returns:
            dict: A dictionary containing Accuracy, F1, Precision, and Recall.
        """
        # Assuming the dataset is all human (class 0) or AI (class 1)
        true_labels = [data_label] * len(classified_dataset)  # All human (class 0)
        
        # Extract predicted labels from the 'cls_score' column
        predicted_labels = classified_dataset['cls_score']

        # Calculate accuracy
        accuracy = accuracy_score(true_labels, predicted_labels)

        # Calculate F1 score
        f1 = f1_score(true_labels, predicted_labels)

        # Calculate Precision
        precision = precision_score(true_labels, predicted_labels, zero_division=1)

        # Calculate Recall
        recall = recall_score(true_labels, predicted_labels, zero_division=1)

        # Return the metrics as a dictionary
        return {
            'detector_accuracy': accuracy,
            'detector_f1_score': f1,
            'detector_precision': precision,
            'detector_recall': recall
        }