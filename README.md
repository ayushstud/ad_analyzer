# AdLens - AI Ad Analyzer

An AI-powered ad analysis tool that combines **multiple ML algorithms** with the **BLIP vision-language model** to score advertisements and generate actionable improvement suggestions.

## Features

- **Multi-algorithm ML scoring** - Trains 6 ML models (Random Forest, Gradient Boosting, SVR, KNN, Ridge, MLP Neural Network) and builds a voting ensemble of the top 3 for robust predictions.
- **BLIP image analysis** - Upload an ad image for automated captioning, visual clarity scoring, emotional impact detection, and text density analysis.
- **Hybrid scoring** - Combines ML predictions (60%) with VLM visual scores (40%) when an image is provided.
- **Actionable suggestions** - Platform-specific, audience-aware, and timing-based optimization tips.
- **Model transparency** - Displays cross-validation MAE for every trained algorithm so you can see which model performs best.

## Project Structure

```
ad_analyzer/
├── app.py               # Flask routes and request handling
├── ml_engine.py          # ML training pipeline (dataset, models, ensemble)
├── image_analyzer.py     # BLIP vision-language model integration
├── templates/
│   └── index.html        # Frontend UI template
├── uploads/              # Uploaded ad images (auto-created)
├── requirements.txt      # Python dependencies
└── README.md
```

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/ayushstud/ad_analyzer.git
cd ad_analyzer

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
python app.py
```

Open [http://localhost:5000](http://localhost:5000) in your browser.

## ML Algorithms

| Algorithm | Description |
|-----------|-------------|
| Random Forest | Ensemble of 120 decision trees (max depth 8) |
| Gradient Boosting | 150 boosted trees with learning rate 0.1 |
| SVR | Support Vector Regression with RBF kernel |
| KNN | Distance-weighted K-Nearest Neighbors (k=7) |
| Ridge | L2-regularized linear regression |
| MLP | Neural network with 64-32 hidden layers |

All models are evaluated via 5-fold cross-validation. The top 3 are combined into a **VotingRegressor** ensemble for final predictions.

## Input Features

| Feature | Values |
|---------|--------|
| Headline length | Word count of the ad headline |
| Ad type | 1=Poster, 2=Social, 3=Banner, 4=Micro Website, 5=Email |
| Color style | 3=Dark, 6=Balanced, 9=Bright |
| Audience | 1=Gen Z/Students, 2=B2B, 3=General |
| Platform | 1=Instagram, 2=YouTube, 3=Facebook |
| Posting hour | 0-23 |
| Posting day | 1=Mon - 7=Sun |

## License

MIT
