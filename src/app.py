from mylibs import * 

app = Flask(__name__)


#-------- ROUTES GO HERE -----------#

# This method takes input via an HTML page


@app.route('/thoughts')
def thoughts():
   return render_template('thoughts.html')

@app.route('/thoughts-examples/<case>')
def case_study(case):

    url = str(case) + '.html'
    try:
        return render_template(url)
    except:
        return 'Report for \'' + case + '\' was not prepared, please conduct a new search'

@app.route('/thoughts-reports', methods=['POST','GET'])
def result():
    '''Gets prediction using the HTML form'''
    if request.method == 'POST':
        query = request.form['query']       
        q = 'Reports for query: ' + '\'' + str(query).upper() + '\''


    lim = 10000
    years = 50
    mc_steps = 50000
    today = date.today().year
    since = today - years

    add_stops = add_stopwords(query)
    auth_nums_df, auth_cits_df, year_nums_df, year_text = get_api_data(query,lim,since)
  
    authors_plot = plot_authors(auth_nums_df, auth_cits_df, howmany=-9, w=10, h=3.3)

    years_plot = plot_year_nums(query, lim, years, year_nums_df, mc_steps, w=10.0, h=3.4)

    common_words_df = get_top_words(year_text, add_stops, n_words=15)   
    words_plot = plot_top_words(common_words_df[-40:-1], w=10, h=3.6)

    year_text_df = make_df(year_text) 

    title1 = str(since) + '-' + str(since+9)
    cloud1 = show_wordcloud(year_text_df[:10].counts, add_stops, title=title1, w=6)

    title2 = str(today-10) + '-' + str(today-1)
    cloud2 = show_wordcloud(year_text_df[-10:-1].counts, add_stops, title=title2, w=6)
    
    title3 = str(today) + '-' + str(today+5) + ' Projection'
    bag = projected_bag(common_words_df)
    cloud3 = show_projected_wordcloud(bag, year_text, add_stops, title=title3, w=6)


    return(render_template('thoughts-reports.html', 
        results=q, since=since, today=today,
        authors_plot=authors_plot,
        years_plot=years_plot,
        words_plot=words_plot,
        cloud1=cloud1, cloud2=cloud2, cloud3=cloud3))


if __name__ == '__main__':
    app.run()

