# Stock Price Prediction

## StockPrice Prediction is a Deep Learning Based Project, that aims to predict the prices of top 200 Stocks present in BSE through an interactive dashboard, which features Company Selection, A Historical Candle Stick Plot (Also Interactive) and to recent news articles on the company.


1. I used LSTM a Deep Learning Neural Network to train the model
2. However, to make model fairly closer to real value, there's a model for every stock present in top 200 stocks of, NSE (National Stock Exchange)
3. The entire project is showcased via a dashboard which not only gives predictions for the day but also, with a price history of the ticker in beautifully represented and interactive graph. Apart from this the dashboard also features a webview to a prominent news website that displays relevant news with respect to the organisation helping investors in making an informed decision.

However accuracy is a still concern as there are far more variables in predicting stock market share values apart from timeseries.
For example at the time of training the model the model had an RMSE error of maximum INR 10, which means for a share of Reliance Industries priced at approximately 2000 the prediction can go off by just 10 Rupees or even less.
But then after today's union budget the RMSE error might have increased by atleast 10% from previous error.

The Error Ranges from INR 0.4 to INR 10 (depending on the model trained for a ticker)

To address this issue there's only one solution, which is to keep retraining the model periodically, say after each week.
I will keep working on enhancing the product with even more features such as which of the famous Investors are currently invested in the stock or which of the famous investor has sold their shares, to make an even more informed decision for other investors.
<br>
<br>
<br>
The First Phase of the project included
1. Fetching of Data from Web using yfinance package and presenting them in a tabulated form using Pandas
2. Second phase included preprocessing, scaling and train-test split of data as required by the LSTM Module, I've chosen to create the training testing data by dividing the data into sub-matrices of each date's closing price and previous 60 days closing price
3. The same approach mentioned in 2. is followed for each date since 2012.
4. Then the model is trained and evaluated for efficiency and is repeated for all 200 stocks,
5. Then all these individual models are exported
6. <b> The Github Repo Picks Up From there </b>

<a href="https://colab.research.google.com/drive/1w-kPcttT276eflLbBXUGc4FBWONtxqGc?usp=sharing"><b>Click Here to Visit the Colab Notebook that covers Phase 1</b></a>

### Finally, the project is given a graphical interface using Streamlit (in ui.py) that fetches the pretrained model from models folder and predicts price for selected stock

## The project in its full form is deployed on HuggingFaceHub, Huge Credits to them for providing a highly capable and hardware rich system that helps my extremely hardware intensive deeplearning project to run smoothly

## <a href="https://huggingface.co/spaces/tanaysk/stockpricepred"> Click Here to Visit the Deployed Project </a>
