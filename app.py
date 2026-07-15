import streamlit as st
import pandas as pd
import pickle
import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


st.title("Book Recommender System")
df = pd.read_csv('cleaned_data.csv')

if os.path.exists("similarity.pkl"):
    if st.button("Generate similarities"):
        cv = CountVectorizer(max_features = 10000, stop_words = 'english')
        vector = cv.fit_transform(df['tags'])
        vector_df=pd.DataFrame(data=vector.toarray(),columns=cv.get_feature_names_out)      
        similarity = cosine_similarity(vector_df)
        pkl.dump(top_similar, open('similarity.pkl', 'wb'))


with open("similarity.pkl", "rb") as file:
    similarities = pickle.load(file)

movies = df['title'].tolist()

name = st.selectbox("Select a Book", Book)

#Lets write function get name of movie by index
def get_name_by_index(i):
    if i < len(df) and i>0:
        return df.loc[i,'title']
    else:
        return''

def get_index_from_name(name):
    # Normalize user input: lowercase it and strip out all spaces and hyphens
    clean_user_name = name.strip().lower().replace(' ', '').replace('-', '')
    
    # Vectorized pandas match: normalize the dataframe column for comparison
    match = df[df['title'].str.lower().str.replace(' ', '').str.replace('-', '') == clean_user_name]
    
    if not match.empty:
        return match.index[0]
    return -1


if st.button("Recommend Book"):
    index = get_index_from_name(name)
    if index == -1:
        st.write("Book not found.")
    else:
        st.write(f"Recommendations for '{title}' will be displayed here.")
        st.write(f"book index is { index }")
        similarity_indexes = list(enumerate(similarities[index]))
        similarity_indexes = sorted(similarity_indexes, key=lambda x: x[1], reverse=True)
        for i in range(1, 6):
            st.write(i, ":", get_name_by_index(similarity_indexes[i][0]))