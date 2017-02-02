## Evolution of Thoughts from Scholarly Resources

Link to app [demo](http://www.hieuhpham.com/thoughts)


<p style="text-align:justify;"> Provided the search query or topic of interests as an user input, the app will perform the data and text mining from <a href ="http://www.crossref.org/">CrossRef</a> using its REST API. Firstly, it will collect the metadata of 10000 most relevant articles (including their titles, publication date, author list and citation records). Useful statistics and learning will then be extracted based on this data mining. Specifically, the revolution year (i.e. time when publication status changes its bahavior) is captured by MCMC simulation (Markov-Chain Monte Carlo) and the future performance will be predicted with Time-Series ARIMA model (Autoregressive Integrated Moving Average). <p>

  <p style="text-align:justify;"> At the next level, natural language processing (NLP) will be used to analyze the text, composed of the article titles, in order to explore the evolution of concepts within that particular topic over the course of the previous decades. Finally the outstanding trend for upcoming years will be projected (by ARIMA model on most frequent vocabulary).</p>


![workflow] (workflow.png)

Source code for the app is stored in the src folder. Details can also be found in the ipython notebook.
