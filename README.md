# tweet-sentiment-analysis
Clone the project. <br> 
Use `python v3.9`<br> 
Run `pip install -r requirements.txt` <br>
Run `pip install -r requirements.txt jupyter` if you want to use jupyter <br>

## Running Cleaning and get cleaned Twitter/X tweets dataset
Create a `cleaned_data` directory in the main project `tweet-sentiment-analysis` directory

### Jupyter Notebook
1. Start jupyter: `jupyter notebook` or `jupyter lab` in the main project directory
2. Navigate to `cleaning/` directory
3. Run the notebooks in Jupyter Notebook/Lab in the `cleaning/` directory individually. Make sure that Spark is reading the correct file path that leads to the `twitter_training.csv` in the `data/` directory.
   
### Python Script
1. Navigate to the `cleaning/` directory
2. Run `python3 <cleaning method file>.py` in the `cleaning/` directory. Make sure that Spark is reading the correct file path that leads to the `twitter_training.csv` in the `data/` directory.

## Running Evaluation
Double check and make sure that correct cleaned tweet dataset CSV file is created in the `cleaned_data` directory.
Each one should output the evalution and analysis (accuracy, F1 scoring, classification report) for each cleaning method.

File names to look for:<br>
Traditional/Extensive Cleaning: `nlp_data.csv`<br>
Social-media-aware Cleaning: `social_media_cleaned_tweets.csv`<br>
Baseline (No cleaning): `baseline_data.csv`<br>

### Jupyter Notebook
1. Start jupyter: `jupyter notebook` or `jupyter lab` in the main project directory (if you haven't already)
2. Navigate to `testing/` directory
3. Run the notebooks in Jupyter Notebook/Lab in the `testing/` directory individually. Make sure that Spark is reading the correct file path that leads to the corresponding `.csv` in the `cleaned_data/` directory.

Corresponding `.csv` and notebook: <br> 
Traditional/Extensive Cleaning: `nlp_data.csv` --> nlp_evaluation.ipynb<br>
Social-media-aware Cleaning: `social_media_cleaned_tweets.csv` --> sm_evalution.ipynb <br>
Baseline (No cleaning): `baseline_data.csv` --> noclean_evaluation.ipynb<br>

### Python Script
1. Navigate to the `testing/` directory
2. Run `python3 <cleaning method evaluation file>.py` in the `testing/` directory. Make sure that Spark is reading the correct file path that leads to the corresponding `.csv` in the `cleaned_data/` directory.

Corresponding `.csv` and Python script: <br> 
Traditional/Extensive Cleaning: `nlp_data.csv` --> nlp_evaluation.py<br>
Social-media-aware Cleaning: `social_media_cleaned_tweets.csv` --> sm_evalution.py <br>
Baseline (No cleaning): `baseline_data.csv` --> noclean_evaluation.py<br>
