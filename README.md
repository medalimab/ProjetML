# MRI Ligament Analysis App

This Streamlit application provides a user interface for analyzing knee MRI scans to detect:
- ACL tears
- Meniscus tears 
- General abnormalities

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure your trained models are saved in the `./model` directory as:
- `acl_model.h5` - Model for ACL tear detection
- `meniscus_model.h5` - Model for meniscus tear detection  
- `abnormal_model.h5` - Model for general abnormality detection

3. Train the models:
```bash
python train_models.py
```

4. Run the Streamlit app:
```bash
streamlit run app.py
```

## Usage

1. Upload an MRI scan using the file uploader
2. The system will automatically analyze the image
3. Results show probability scores for each condition
4. Higher percentages indicate higher likelihood of the condition

## Data

The models were trained on a dataset of over 1000 MRI scans with expert annotations. Training and validation data statistics are displayed in the sidebar.

## Model Performance

Current validation metrics:
- Accuracy: 85.2%
- Precision: 83.7%
- Recall: 86.9%
