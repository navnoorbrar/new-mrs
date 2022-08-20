import streamlit as st
import json
from classifier import KNearestNeighbours
from operator import itemgetter

#Load data and movies list from corresponding JSON Files
with open(r'newdata.json', 'r+', encoding='utf-8') as f:
    data = json.load(f) #oject is f and through f we are loading data
with open(r'newtitles.json','r+',encoding='utf-8') as f:
    movie_titles = json.load(f)

#Now we will create a method to apply algorithm
def knn(test_point, k): #test_point is our data
    #Create dummy target variable for the KNN Classifier
    target = [0 for item in movie_titles] #beacuse there is a lot of categorical or string data
    # set item as 0, dummy variable is always n-1 so if we set it as 0 1 will automatically be selected
    #Instantiate object for the classifier
    model = KNearestNeighbours(data, target, test_point, k=k)
    #Run the algorithm and find the distance
    model.fit()
    #Distances to most distant movie
    max_dist = sorted(model.distances, key=itemgetter(0))[-1]
    #Print list of 10 recommendations < change value of k for a different number >
    table = list()
    for i in model.indices: #by using for loop we called the indeces
        # Returns back movie title and imbd link
        table.append([movie_titles[i][0], movie_titles[i][2]])#added data at 0th and 2nd index of movie_titles
    return table
#Now we will run our main application
if __name__=='__main__':
     genres =  ['Action','Adventure','Animation','Biography','Comedy','Crime','Documentary','Drama','Family',
             'Fantasy','Film-Noir','Game-Show','History','Horror','Music','Musical','Mystry','News',
             'Reality-TV','Romance','Sci-Fi','Short','Sport','Thriller','War','Western']

     movies = [title[0] for title in movie_titles]
     st.header('Movie Recommendation System')  #This is heading of our application
     # Now we will add option movie based or genre based
     apps = ['--Select--', 'Movie based', 'Genres based']
     app_options = st.selectbox('Select application:', apps)
     #below code means if the app option is movie based then give option of slecting movies
     if app_options == 'Movie based':
         movie_select = st.selectbox('Select movie:', ['--Select--'] + movies)
         if movie_select == '--Select--':
             st.write('Select a movie')
         else:
             n = st.number_input('Number of movies:', min_value=5, max_value=20, step=1)
             genres = data[movies.index(movie_select)]
             test_point = genres
             table = knn(test_point, n)
             for movie, link in table:
                 # Displays movie title with link to imdb
                 st.markdown(f"[{movie}]({link})")
     elif app_options == apps[2]:
         options = st.multiselect('Select genres:', genres)
         if options: #if genres is available then
             imdb_score = st.slider('IMDB score:',1,10,8) #Rating is form 1 to 10 and keepm default value as 8
             n = st.number_input('Number of movies:', min_value=5, max_value=20, step=1)
             test_point = [1 if genre in options else 0 for genre in genres]
             test_point.append(imdb_score)
             table = knn(test_point, n)
             for movie, link in table:
                 #Displays movie title with link to imdb
                 st.markdown(f"[{movie}]({link})")
         else:
             st.write("This is a simple Movie Recommender application. "
                      "You can select the genres and change the IMDB score")

     else:
         st.write('Select option')