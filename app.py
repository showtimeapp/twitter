import pandas as pd
import numpy as np
import streamlit as st
from langchain_groq import ChatGroq
from langchain.schema import SystemMessage, HumanMessage
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from sklearn.feature_extraction.text import TfidfVectorizer
from langchain.embeddings.base import Embeddings
import os
from typing import Dict, List, Any
import plotly.express as px
import plotly.graph_objects as go
import wikipedia
import re
from difflib import get_close_matches
import requests
import json
from fpdf import FPDF
import base64
import io
from datetime import datetime

# Set fixed Groq API key
os.environ["GROQ_API_KEY"] = "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"

class ReportPDF(FPDF):
    """Custom PDF class with header and footer"""
    def header(self):
        # Set font for header
        self.set_font('Arial', 'B', 12)
        # Title
        self.cell(0, 10, 'Twitter Analysis Report', 0, 1, 'C')
        # Line break
        self.ln(5)

    def footer(self):
        # Position cursor at 1.5 cm from bottom
        self.set_y(-15)
        # Set font for footer
        self.set_font('Arial', 'I', 8)
        # Page number
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', 0, 0, 'C')
        # Add date
        date_str = datetime.now().strftime("%Y-%m-%d")
        self.cell(0, 10, date_str, 0, 0, 'R')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 14)
        self.set_fill_color(200, 220, 255)
        self.cell(0, 10, title, 0, 1, 'L', 1)
        self.ln(5)

    def chapter_body(self, text):
        self.set_font('Arial', '', 11)
        # Process text to properly handle paragraphs
        text = text.encode('latin-1', 'replace').decode('latin-1')
        self.multi_cell(0, 5, text)
        self.ln(5)

    def section_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 8, title, 0, 1, 'L')
        self.ln(2)

    def info_row(self, label, value):
        self.set_font('Arial', 'B', 11)
        self.cell(50, 6, label, 0, 0, 'L')
        self.set_font('Arial', '', 11)
        self.multi_cell(0, 6, value)

class SimpleEmbeddings(Embeddings):
    """A simple embedding class using TF-IDF."""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.fitted = False
        self.vectors = None
        self.documents = None
    
    def embed_documents(self, texts):
        """Embed a list of documents using TF-IDF."""
        if not self.fitted:
            self.vectors = self.vectorizer.fit_transform(texts)
            self.fitted = True
            self.documents = texts
        else:
            self.vectors = self.vectorizer.transform(texts)
            self.documents = texts
        
        # Convert sparse matrix to dense array
        return self.vectors.toarray()
    
    def embed_query(self, text):
        """Embed a query using TF-IDF."""
        # If not fitted, return zeros
        if not self.fitted:
            return np.zeros(100)
        
        vector = self.vectorizer.transform([text])
        return vector.toarray()[0]
    
class TwitterAnalyzer:
    def __init__(self, df):
        """
        Initialize the Twitter analyzer with a DataFrame.
        
        Args:
            df: DataFrame containing Twitter data
        """
        # Store the data
        self.df = df
        
        # Convert numeric columns to appropriate types
        numeric_columns = ['likes', 'replies', 'retweets']
        for col in numeric_columns:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        # Initialize the Groq LLM client
        self.llm = ChatGroq(
            model_name="llama3-8b-8192",
            temperature=0.2,
            groq_api_key=os.environ["GROQ_API_KEY"]
        )
        
        # Initialize embedding model
        self.embeddings = SimpleEmbeddings()
        
        # Initialize memory for conversation context
        self.memory = ConversationBufferMemory()
        
        # Initialize vector database for RAG
        self._initialize_vector_store()
        
        # Store Wikipedia data cache
        self.wiki_cache = {}
        
        # Store Gemini news cache
        self.news_cache = {}
        
        # Gemini configuration with fixed API key
        self.gemini_api_key = "AIzaSyBHBFb7iJ4VNSazi_oNZ50pwSo8suH7Y4M"
        self.gemini_model = "models/gemini-2.0-flash"

    def get_news_with_gemini_search(self, query):
        """
        Get latest news about a topic using Gemini's search capability.
        
        Args:
            query: The query to search for
            
        Returns:
            News content as string, or None if an error occurs
        """
        # Check if we already have this query cached
        if query in self.news_cache:
            return self.news_cache[query]
            
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/{self.gemini_model}:generateContent?key={self.gemini_api_key}"
            headers = {
                "Content-Type": "application/json"
            }
            payload = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": f"Get detailed and latest news related to: {query}. Include sources if available."}]
                    }
                ],
                "tools": [
                    {
                        "googleSearch": {}
                    }
                ]
            }
            
            response = requests.post(url, headers=headers, json=payload)
            
            if response.status_code == 200:
                data = response.json()
                news_content = data['candidates'][0]['content']['parts'][0]['text']
                
                # Cache the result
                self.news_cache[query] = news_content
                return news_content
            else:
                print(f"Gemini API Error: {response.text}")
                return None
        except Exception as e:
            print(f"Error fetching news: {str(e)}")
            return None

    def get_with_gemini_search(self, query):
        """
        Get latest news about a topic using Gemini's search capability.
        
        Args:
            query: The query to search for
            
        Returns:
            News content as string, or None if an error occurs
        """
        # Check if we already have this query cached
        if query in self.news_cache:
            return self.news_cache[query]
            
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/{self.gemini_model}:generateContent?key={self.gemini_api_key}"
            headers = {
                "Content-Type": "application/json"
            }
            payload = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": f"{query}."}]
                    }
                ],
                "tools": [
                    {
                        "googleSearch": {}
                    }
                ]
            }
            
            response = requests.post(url, headers=headers, json=payload)
            
            if response.status_code == 200:
                data = response.json()
                news_content = data['candidates'][0]['content']['parts'][0]['text']
                
                # Cache the result
                self.news_cache[query] = news_content
                return news_content
            else:
                print(f"Gemini API Error: {response.text}")
                return None
        except Exception as e:
            print(f"Error fetching news: {str(e)}")
            return None
        
    def get_wikipedia_data(self, entity_name):
        """
        Fetch data about an entity from Wikipedia.
        
        Args:
            entity_name: Name of the entity to search for
            
        Returns:
            Dictionary containing title and content, or None if not found
        """
        # Check if we already fetched this entity
        if entity_name in self.wiki_cache:
            return self.wiki_cache[entity_name]
        
        try:
            # Try to get the Wikipedia page
            page = wikipedia.page(entity_name, auto_suggest=True)
            wiki_data = {
                "title": page.title,
                "content": page.content,
                "url": page.url
            }
            
            # Cache the result
            self.wiki_cache[entity_name] = wiki_data
            return wiki_data
            
        except wikipedia.exceptions.DisambiguationError as e:
            # Handle disambiguation by taking the first option
            try:
                if e.options:
                    page = wikipedia.page(e.options[0], auto_suggest=False)
                    wiki_data = {
                        "title": page.title,
                        "content": page.content,
                        "url": page.url
                    }
                    self.wiki_cache[entity_name] = wiki_data
                    return wiki_data
            except:
                pass
            return None
            
        except wikipedia.exceptions.PageError:
            # Page not found
            return None
            
        except Exception as e:
            # Other errors
            print(f"Wikipedia error for {entity_name}: {str(e)}")
            return None

    def extract_entities(self, question):
        """
        Extract potential named entities from a question.
        
        Args:
            question: The question text
            
        Returns:
            List of potential entity names
        """
        # First, check if any user in our dataset is mentioned
        user_names = self.df['fullname'].unique()
        mentioned_users = []
        
        for user in user_names:
            if user.lower() in question.lower():
                mentioned_users.append(user)
        
        # If we found users in our dataset, return them
        if mentioned_users:
            return mentioned_users
        
        # Otherwise, try to extract other potential entities
        # This is a simple approach - in production, you might use NER models
        # Look for capitalized phrases which might be names
        potential_entities = re.findall(r'([A-Z][a-z]+(?: [A-Z][a-z]+)*)', question)
        
        # Filter out common words that might be capitalized
        common_words = ["what", "who", "where", "when", "why", "how", "is", "are", "do", "does", 
                        "did", "can", "could", "would", "should", "may", "might", "must", "the"]
        
        filtered_entities = []
        for entity in potential_entities:
            if entity.lower() not in common_words and len(entity) > 3:
                filtered_entities.append(entity)
                
        return filtered_entities
        
    def _initialize_vector_store(self):
        """
        Initialize the vector store for RAG with tweet content.
        """
        # Prepare documents for the vector store
        documents = []
        
        # Create documents with rich metadata
        for idx, row in self.df.iterrows():
            # Create a document for each tweet with all relevant metadata
            document_text = f"{row['tweet']}"
            document_metadata = {
                "fullname": row["fullname"],
                "likes": int(row["likes"]),
                "replies": int(row["replies"]),
                "retweets": int(row["retweets"]),
                "tweet_idx": idx
            }
            documents.append({"content": document_text, "metadata": document_metadata})
        
        # Split documents for better retrieval (if needed)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        
        # Create vector store index
        try:
            texts = [doc["content"] for doc in documents]
            metadatas = [doc["metadata"] for doc in documents]
            
            self.vector_store = FAISS.from_texts(
                texts=texts,
                embedding=self.embeddings,
                metadatas=metadatas
            )
            st.session_state['vectorstore_initialized'] = True
        except Exception as e:
            st.error(f"Error initializing vector store: {str(e)}")
            self.vector_store = None
            st.session_state['vectorstore_initialized'] = False
            
    def get_data_schema(self) -> str:
        """
        Return the schema of the loaded data.
        """
        return str(self.df.dtypes)
    
    def get_data_summary(self) -> Dict[str, Any]:
        """
        Return basic statistics about the data.
        """
        return {
            "total_tweets": len(self.df),
            "unique_users": self.df['fullname'].nunique(),
            "user_list": self.df['fullname'].unique().tolist(),
            "avg_likes": self.df['likes'].mean(),
            "avg_replies": self.df['replies'].mean(),
            "avg_retweets": self.df['retweets'].mean()
        }
    
    def answer_statistical_question(self, question: str) -> str:
        """
        Answer statistical questions about the data using NumPy and Pandas.
        
        Args:
            question: The statistical question to answer
        
        Returns:
            The answer to the statistical question
        """
        # Pre-process common statistical questions
        question_lower = question.lower()
        
        # Get user-specific data if a name is mentioned
        user_names = self.df['fullname'].unique()
        mentioned_user = None
        for user in user_names:
            if user.lower() in question_lower:
                mentioned_user = user
                break
        
        try:
            # Handle different types of statistical questions
            if "total sum of likes" in question_lower or "sum of likes" in question_lower:
                if mentioned_user:
                    user_df = self.df[self.df['fullname'] == mentioned_user]
                    result = user_df['likes'].sum()
                    return f"The total sum of likes for {mentioned_user} is {result}."
                else:
                    result = self.df['likes'].sum()
                    return f"The total sum of likes across all tweets is {result}."
                    
            elif "total sum of replies" in question_lower or "sum of replies" in question_lower:
                if mentioned_user:
                    user_df = self.df[self.df['fullname'] == mentioned_user]
                    result = user_df['replies'].sum()
                    return f"The total sum of replies for {mentioned_user} is {result}."
                else:
                    result = self.df['replies'].sum()
                    return f"The total sum of replies across all tweets is {result}."
                    
            elif "total sum of retweets" in question_lower or "sum of retweets" in question_lower:
                if mentioned_user:
                    user_df = self.df[self.df['fullname'] == mentioned_user]
                    result = user_df['retweets'].sum()
                    return f"The total sum of retweets for {mentioned_user} is {result}."
                else:
                    result = self.df['retweets'].sum()
                    return f"The total sum of retweets across all tweets is {result}."
                    
            elif "highest number of likes" in question_lower or "most likes" in question_lower:
                max_likes_row = self.df.loc[self.df['likes'].idxmax()]
                return f"The tweet with the highest number of likes ({max_likes_row['likes']}) is by {max_likes_row['fullname']}: '{max_likes_row['tweet']}'."
                
            elif "highest number of replies" in question_lower or "most replies" in question_lower:
                max_replies_row = self.df.loc[self.df['replies'].idxmax()]
                return f"The tweet with the highest number of replies ({max_replies_row['replies']}) is by {max_replies_row['fullname']}: '{max_replies_row['tweet']}'."
                
            elif "highest number of retweets" in question_lower or "most retweets" in question_lower:
                max_retweets_row = self.df.loc[self.df['retweets'].idxmax()]
                return f"The tweet with the highest number of retweets ({max_retweets_row['retweets']}) is by {max_retweets_row['fullname']}: '{max_retweets_row['tweet']}'."
                
            elif "average" in question_lower or "mean" in question_lower:
                if "likes" in question_lower:
                    if mentioned_user:
                        user_df = self.df[self.df['fullname'] == mentioned_user]
                        result = user_df['likes'].mean()
                        return f"The average number of likes for {mentioned_user} is {result:.2f}."
                    else:
                        result = self.df['likes'].mean()
                        return f"The average number of likes across all tweets is {result:.2f}."
                elif "replies" in question_lower:
                    if mentioned_user:
                        user_df = self.df[self.df['fullname'] == mentioned_user]
                        result = user_df['replies'].mean()
                        return f"The average number of replies for {mentioned_user} is {result:.2f}."
                    else:
                        result = self.df['replies'].mean()
                        return f"The average number of replies across all tweets is {result:.2f}."
                elif "retweets" in question_lower:
                    if mentioned_user:
                        user_df = self.df[self.df['fullname'] == mentioned_user]
                        result = user_df['retweets'].mean()
                        return f"The average number of retweets for {mentioned_user} is {result:.2f}."
                    else:
                        result = self.df['retweets'].mean()
                        return f"The average number of retweets across all tweets is {result:.2f}."
            
            # For more complex statistical questions, delegate to the semantic QA system
            return self.answer_semantic_question(f"Statistical analysis: {question}")
            
        except Exception as e:
            return f"Error processing statistical question: {str(e)}"
    
    def answer_semantic_question(self, question: str) -> str:
        """
        Answer semantic questions about tweet content using Groq LLM with RAG,
        enhanced with Wikipedia data and latest news.
        
        Args:
            question: The semantic question to answer
        
        Returns:
            The LLM's answer to the semantic question
        """
        # Format the question for better RAG performance
        question_lower = question.lower()
        
        # Check if vector store is initialized
        if not self.vector_store:
            return "Unable to perform semantic analysis. Vector store is not initialized."
            
        # Extract potential entities from the question
        entities = self.extract_entities(question)
        
        # Get Wikipedia data for mentioned entities
        wiki_context = ""
        for entity in entities:
            wiki_data = self.get_wikipedia_data(entity)
            if wiki_data:
                # Extract only the first few paragraphs to keep context manageable
                paragraphs = wiki_data["content"].split("\n\n")
                intro = "\n\n".join(paragraphs[:3])  # First three paragraphs
                
                wiki_context += f"\nWikipedia information about {wiki_data['title']}:\n"
                wiki_context += f"{intro}\n"
                # wiki_context += f"Source: {wiki_data['url']}\n\n"
        
        # # Get news context for entities and their relationships
        # news_context = ""
        # # First try to get news for individual entities
        # for entity in entities:
        #     news = self.get_news_with_gemini_search(entity)
        #     if news:
        #         news_context += f"\nLatest news about {entity}:\n{news}\n\n"
        
        # # If the question seems to be about a relationship between entities, search for that
        # if len(entities) >= 2 and any(term in question_lower for term in ["relation", "connection", "about", "against", "between", "and"]):
        #     relationship_query = " and ".join(entities[:2])  # Take first two entities
        #     news = self.get_news_with_gemini_search(relationship_query)
        #     if news:
        #         news_context += f"\nLatest news about {relationship_query}:\n{news}\n\n"
        
        # # If the question itself seems focused, use it directly as a news query
        # specific_terms = ["why", "how", "what happened", "incident", "event", "scandal", "controversy"]
        # if any(term in question_lower for term in specific_terms):
        #     news = self.get_news_with_gemini_search(question)
        #     if news:
        #         news_context += f"\nLatest news related to the question:\n{news}\n\n"
        
        # Get user-specific mentions for more targeted retrieval
        user_names = self.df['fullname'].unique()
        mentioned_users = []
        for user in user_names:
            if user.lower() in question_lower:
                mentioned_users.append(user)
        
        try:
            # Retrieve relevant documents from vector store
            relevant_docs = []
            
            # Semantic search specific to mentioned users or general if none mentioned
            if mentioned_users:
                combined_results = []
                # For each mentioned user, perform a separate search
                for user in mentioned_users:
                    # Add filter to search only tweets from this user
                    filtered_docs = self.vector_store.similarity_search(
                        question,
                        k=5,
                        filter={"fullname": user}
                    )
                    combined_results.extend(filtered_docs)
                
                # If looking for comparison, ensure we have content from both users
                if "compare" in question_lower and len(mentioned_users) >= 2:
                    for user in mentioned_users[:2]:
                        # Ensure we have at least 3 tweets from each user for comparison
                        user_docs = [doc for doc in combined_results if doc.metadata["fullname"] == user]
                        if len(user_docs) < 3:
                            # Add more tweets from this user
                            more_docs = self.vector_store.similarity_search(
                                "representative tweets",
                                k=3-len(user_docs),
                                filter={"fullname": user}
                            )
                            combined_results.extend(more_docs)
                
                relevant_docs = combined_results
            else:
                # General search for all tweets
                relevant_docs = self.vector_store.similarity_search(question, k=8)
            
            # Format retrieved documents for context
            context = ""
            
            # Create richer context with categorization
            tweet_contexts = {}
            for doc in relevant_docs:
                user = doc.metadata["fullname"]
                if user not in tweet_contexts:
                    tweet_contexts[user] = []
                
                tweet_info = {
                    "text": doc.page_content,
                    "likes": doc.metadata["likes"],
                    "replies": doc.metadata["replies"],
                    "retweets": doc.metadata["retweets"]
                }
                tweet_contexts[user].append(tweet_info)
            
            # Format context by user
            for user, tweets in tweet_contexts.items():
                context += f"\nTweets by {user}:\n"
                for i, tweet in enumerate(tweets, 1):
                    context += f"{i}. \"{tweet['text']}\" (Likes: {tweet['likes']}, Replies: {tweet['replies']}, Retweets: {tweet['retweets']})\n"
            
            # Add statistical context to enhance semantic understanding
            if mentioned_users:
                context += "\nStatistical Summary:\n"
                for user in mentioned_users:
                    user_df = self.df[self.df['fullname'] == user]
                    context += f"{user} statistics: {len(user_df)} tweets, "
                    context += f"Average likes: {user_df['likes'].mean():.1f}, "
                    context += f"Average replies: {user_df['replies'].mean():.1f}, "
                    context += f"Average retweets: {user_df['retweets'].mean():.1f}\n"
            
            # Add comparison data for comparison questions
            if "compare" in question_lower and len(mentioned_users) >= 2:
                user1, user2 = mentioned_users[:2]
                user1_df = self.df[self.df['fullname'] == user1]
                user2_df = self.df[self.df['fullname'] == user2]
                
                context += f"\nDetailed Comparison between {user1} and {user2}:\n"
                context += f"- Tweet count: {user1}: {len(user1_df)} vs {user2}: {len(user2_df)}\n"
                context += f"- Average likes: {user1}: {user1_df['likes'].mean():.1f} vs {user2}: {user2_df['likes'].mean():.1f}\n"
                context += f"- Average replies: {user1}: {user1_df['replies'].mean():.1f} vs {user2}: {user2_df['replies'].mean():.1f}\n"
                context += f"- Average retweets: {user1}: {user1_df['retweets'].mean():.1f} vs {user2}: {user2_df['retweets'].mean():.1f}\n"
            
            # For theme/focus area questions, include more tweets from the user
            if "main theme" in question_lower or "focus area" in question_lower or "summary" in question_lower:
                if mentioned_users:
                    for user in mentioned_users:
                        user_df = self.df[self.df['fullname'] == user]
                        context += f"\nAll tweets by {user} for theme analysis:\n"
                        for i, (_, row) in enumerate(user_df.iterrows(), 1):
                            context += f"{i}. \"{row['tweet']}\"\n"
            
            # Combine tweet context, Wikipedia context, and news context
            combined_context = context
            if wiki_context:
                combined_context += "\n" + wiki_context
            # if news_context:
            #     combined_context += "\n" + news_context
            
            # Create system prompt with explicit instructions based on question type
            system_prompt = f"""
            You are an AI assistant that performs detailed analysis of Twitter data, 
            enhanced with Wikipedia information when available. Below is the relevant context:
            
            {combined_context}
            
            Guidelines for your analysis:
            1. Base your answer on the provided tweets, statistics, Wikipedia data - do not make up information.
            2. Provide evidence for your claims by directly referencing specific tweets or Wikipedia information.
            3. Use exact quotes when discussing tweet content.
            4. Include relevant engagement metrics (likes, replies, retweets) when they support your analysis.
            5. When Wikipedia information or news is available, incorporate it to provide broader context about people, organizations, or topics mentioned.
            6. If the latest news provides insights into current events related to the question, use that information to contextualize the tweets.
            7. Be concise but thorough in your analysis.
            
            If you're analyzing themes or focus areas:
            - Identify the most common topics across the tweets
            - Look for recurring words, hashtags, or ideas
            - Consider the issues that get the most engagement
            - Connect tweet topics to biographical or background information when available
            - If available, use latest news to see how the tweets relate to current events
            
            If you're comparing users:
            - Highlight differences in content focus, tone, and engagement metrics
            - Note any common interests or diverging opinions between them
            - Use background information to provide context for their positions
            
            If the question relates to statistics, provide precise numbers from the context.
            
            Now, please answer the following question based on the above data and guidelines:
            """
            
            # Create messages
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=question)
            ]
            
            # Get response from LLM
            response = self.llm(messages)
            return response.content
        except Exception as e:
            return f"Error in semantic analysis: {str(e)}"

    def analyze(self, question: str) -> str:
        """
        Main method to analyze questions and route them to appropriate handlers.
        
        Args:
            question: The question to answer
        
        Returns:
            The answer to the question
        """
        question_lower = question.lower()
        
        # Check if the question is about tweet content/themes/comparison
        semantic_keywords = [
            'what', 'theme', 'focus', 'topic', 'about', 'discuss', 'mention', 'say', 'opinion',
            'compare', 'comparison', 'difference', 'similar', 'differ', 'perspective', 'view',
            'summary', 'summarize', 'analyze', 'main', 'point', 'idea', 'subject'
        ]
        
        # Determine if the question is statistical or semantic
        statistical_keywords = [
            'sum', 'total', 'average', 'mean', 'median', 'count', 'highest', 'lowest', 
            'maximum', 'minimum', 'most', 'least', 'percentage', 'number', 'how many'
        ]
        
        is_statistical = any(keyword in question_lower for keyword in statistical_keywords)
        is_semantic = any(keyword in question_lower for keyword in semantic_keywords) or 'tweet' in question_lower
        
        # If both types of keywords appear, determine which is more dominant
        if is_statistical and is_semantic:
            stat_count = sum(1 for keyword in statistical_keywords if keyword in question_lower)
            sem_count = sum(1 for keyword in semantic_keywords if keyword in question_lower)
            is_statistical = stat_count > sem_count
            is_semantic = sem_count >= stat_count
        
        if is_statistical:
            return self.answer_statistical_question(question)
        else:
            return self.answer_semantic_question(question)
    
    def generate_user_statistics_chart(self, username=None):
        """
        Generate charts for user statistics or overall statistics if no username provided.
        """
        if username:
            user_df = self.df[self.df['fullname'] == username]
            if len(user_df) == 0:
                return None, "User not found in dataset"
            
            # Create a bar chart for user's engagement metrics
            fig = go.Figure(data=[
                go.Bar(name='Likes', x=user_df.index, y=user_df['likes'], marker_color='blue'),
                go.Bar(name='Replies', x=user_df.index, y=user_df['replies'], marker_color='green'),
                go.Bar(name='Retweets', x=user_df.index, y=user_df['retweets'], marker_color='red')
            ])
            
            fig.update_layout(
                title=f'Engagement Metrics for {username}',
                xaxis_title='Tweet Index',
                yaxis_title='Count',
                barmode='group',
                height=500
            )
            
            return fig, None
        else:
            # Create aggregated user statistics
            user_stats = self.df.groupby('fullname').agg({
                'likes': 'sum',
                'replies': 'sum',
                'retweets': 'sum',
                'tweet': 'count'
            }).reset_index()
            user_stats.rename(columns={'tweet': 'tweet_count'}, inplace=True)
            
            # Create a figure with subplots
            fig = px.bar(
                user_stats, 
                x='fullname', 
                y=['likes', 'replies', 'retweets'],
                title="Engagement Metrics by User",
                labels={'value': 'Count', 'fullname': 'User', 'variable': 'Metric'},
                height=500,
                barmode='group'
            )
            
            return fig, None
    
    def generate_comparison_chart(self, user1, user2):
        """
        Generate comparison charts between two users.
        """
        if user1 == user2:
            return None, "Please select two different users for comparison"
            
        user1_df = self.df[self.df['fullname'] == user1]
        user2_df = self.df[self.df['fullname'] == user2]
        
        if len(user1_df) == 0 or len(user2_df) == 0:
            return None, "One or both users not found in dataset"
        
        # Calculate average metrics
        user1_avg = user1_df[['likes', 'replies', 'retweets']].mean()
        user2_avg = user2_df[['likes', 'replies', 'retweets']].mean()
        
        # Combine into a DataFrame for plotting
        comparison_df = pd.DataFrame({
            'Metric': ['Likes', 'Replies', 'Retweets'],
            user1: [user1_avg['likes'], user1_avg['replies'], user1_avg['retweets']],
            user2: [user2_avg['likes'], user2_avg['replies'], user2_avg['retweets']]
        })
        
        # Create comparison chart
        fig = px.bar(
            comparison_df, 
            x='Metric', 
            y=[user1, user2],
            title=f"Average Engagement Metrics: {user1} vs {user2}",
            labels={'value': 'Average Count', 'variable': 'User'},
            barmode='group',
            height=500
        )
        
        return fig, None

    def get_news_analysis(self, entities,question: str) -> str:
        """Get news analysis for extracted entities"""
        # news_analysis = []
        # for entity in entities:
        #     news = self.get_news_with_gemini_search(entity)
        #     if news:
        #         news_analysis.append(f"**Latest news about {entity}:**\n{news}")
        question_lower = question.lower()
        # Get news context for entities and their relationships
        news_analysis = []
        # First try to get news for individual entities
        for entity in entities:
            news = self.get_news_with_gemini_search(question)
            if news:
                news_analysis.append(f"\nLatest news related to the question:\n{news}\n\n")
        
        # # If the question seems to be about a relationship between entities, search for that
        # if len(entities) >= 2 and any(term in question_lower for term in ["relation", "connection", "about", "against", "between", "and"]):
        #     relationship_query = " and ".join(entities[:2])  # Take first two entities
        #     news = self.get_news_with_gemini_search(relationship_query)
        #     if news:
        #         news_analysis.append(f"\nLatest news about {relationship_query}:\n{news}\n\n")
        
        # # If the question itself seems focused, use it directly as a news query
        # specific_terms = ["why", "how", "what happened", "incident", "event", "scandal", "controversy"]
        # if any(term in question_lower for term in specific_terms):
        #     news = self.get_news_with_gemini_search(question)
        #     if news:
        #         news_analysis.append(f"\nLatest news related to the question:\n{news}\n\n")
        return "\n\n".join(news_analysis)
    
def generate_person_pdf(analyzer, person_name, question, answer, news_analysis=None):
    """
    Generate a PDF report about a person based on Wikipedia info, news, and tweet analysis.
    
    Args:
        analyzer: TwitterAnalyzer instance
        person_name: Name of the person
        question: The question asked
        answer: The answer generated
        news_analysis: Optional news analysis text
        
    Returns:
        Base64 encoded PDF content
    """
    # Get Wikipedia data
    wiki_data = analyzer.get_wikipedia_data(person_name)
    
    # Get latest news
    news_data = analyzer.get_news_with_gemini_search(person_name)
    
    # Get additional detailed info from Gemini
    detailed_info = analyzer.get_with_gemini_search(
        f"Provide structured information about {person_name} including: "
        "Current Position, Political Affiliation, Election History, "
        "Social Media Presence, Key Statements and Contributions, "
        "Sentiment Summary (Based on Public Statements in the Last 12 Months), "
        "and Engagement Summary (Last 12 Months)."
    )
    
    # Function to clean and format text for PDF
    def clean_text(text):
        if text is None:
            return ""
        
        # Replace problematic Unicode characters with ASCII equivalents
        text = text.replace('\u2013', '-')  # en dash
        text = text.replace('\u2014', '--')  # em dash
        text = text.replace('\u2018', "'")   # left single quote
        text = text.replace('\u2019', "'")   # right single quote
        text = text.replace('\u201c', '"')   # left double quote
        text = text.replace('\u201d', '"')   # right double quote
        text = text.replace('\u2022', '*')   # bullet
        text = text.replace('\u2026', '...') # ellipsis
        
        # Remove Markdown formatting (bold, italics, etc.)
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Remove bold formatting
        text = re.sub(r'\*(.*?)\*', r'\1', text)      # Remove italic formatting
        
        # Use replace error handling to substitute characters that can't be encoded
        return text.encode('latin-1', 'replace').decode('latin-1')
    
    # Function to process detailed information text to extract structured data
    def parse_detailed_info(text):
        if not text:
            return []
        
        # Initialize results
        sections = []
        current_section = None
        current_content = []
        
        # Split text into lines and process
        lines = text.split('\n')
        for line in lines:
            # Clean the line
            line = line.strip()
            if not line:
                continue
                
            # Check if line is a section header (starts with numbers or bold text)
            if re.match(r'^\d+\.|\*\*[^:]+\*\*$', line):
                # If we have a previous section, add it to the results
                if current_section and current_content:
                    sections.append((current_section, '\n'.join(current_content)))
                    current_content = []
                
                # Extract section title (remove bold formatting and numbers)
                current_section = re.sub(r'^\d+\.\s*|\*\*|\*', '', line)
                
            # Check if line is a subsection (key-value pair)
            elif ':' in line and not line.endswith(':'):
                key, value = line.split(':', 1)
                # Remove bold formatting from key
                key = re.sub(r'\*\*|\*', '', key).strip()
                value = re.sub(r'\*\*|\*', '', value).strip()
                
                current_content.append(f"{key}: {value}")
                
            # Otherwise, treat as content for current section
            else:
                # Remove any remaining markdown
                clean_line = re.sub(r'\*\*|\*', '', line)
                current_content.append(clean_line)
        
        # Add the last section if it exists
        if current_section and current_content:
            sections.append((current_section, '\n'.join(current_content)))
            
        return sections
    
    # Create PDF
    pdf = ReportPDF()
    pdf.alias_nb_pages()
    pdf.add_page()
    
    # Title
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, f"Analysis Report: {clean_text(person_name)}", 0, 1, 'C')
    pdf.ln(5)
    
    # Biography section
    pdf.chapter_title("Background")
    if wiki_data:
        # Extract the first paragraph for the brief
        paragraphs = wiki_data["content"].split("\n\n")
        brief = "\n\n".join(paragraphs[:1])  # First two paragraphs
        pdf.chapter_body(clean_text(brief))
    else:
        pdf.chapter_body("No biographical information available.")
    
    # Detailed information section
    pdf.chapter_title("Profile Details")
    try:
        if detailed_info:
            # Parse the detailed info into sections
            sections = parse_detailed_info(detailed_info)
            
            if sections:
                for section_title, section_content in sections:
                    pdf.section_title(section_title)
                    pdf.chapter_body(clean_text(section_content))
            else:
                # Fallback to simple formatting
                # Remove Markdown formatting
                cleaned_info = re.sub(r'\*\*|\*', '', detailed_info)
                pdf.chapter_body(clean_text(cleaned_info))
        else:
            pdf.chapter_body("No detailed profile information available.")
    except Exception as e:
        pdf.chapter_body(f"Error processing profile details: {str(e)}")
    
    # Latest news section
    pdf.chapter_title("Latest News")
    if news_analysis:
        # Process news analysis to remove markdown and improve formatting
        cleaned_news = clean_text(re.sub(r'\*\*([^*]+)\*\*', r'\1', news_analysis))
        
        # Split into sections
        news_sections = re.split(r'(?:\*\*|\n)([A-Z][A-Za-z\s]+:)', cleaned_news)
        
        if len(news_sections) > 1:
            current_section = None
            for i, section in enumerate(news_sections):
                if i == 0:  # First section is intro text
                    if section.strip():
                        pdf.chapter_body(section.strip())
                elif i % 2 == 1:  # Odd indices are section titles
                    current_section = section
                    pdf.section_title(current_section)
                else:  # Even indices are section content
                    pdf.chapter_body(section.strip())
        else:
            pdf.chapter_body(cleaned_news)
    else:
        pdf.chapter_body("No recent news available.")
    
    # Question and answer section
    pdf.chapter_title("Analysis")
    pdf.section_title("Question")
    pdf.chapter_body(clean_text(question))
    
    pdf.section_title("Answer")
    pdf.chapter_body(clean_text(answer))
    
    # Get tweet data specifically about this person
    try:
        user_df = analyzer.df[analyzer.df['fullname'] == person_name]
        if len(user_df) > 0:
            pdf.chapter_title("Tweet Data Analysis")
            pdf.section_title("Tweet Statistics")
            stats_text = f"Total tweets analyzed: {len(user_df)}\n"
            stats_text += f"Average likes per tweet: {user_df['likes'].mean():.1f}\n"
            stats_text += f"Average replies per tweet: {user_df['replies'].mean():.1f}\n"
            stats_text += f"Average retweets per tweet: {user_df['retweets'].mean():.1f}\n"
            pdf.chapter_body(clean_text(stats_text))
            
            pdf.section_title("Recent Tweets")
            for i, (_, row) in enumerate(user_df.head(5).iterrows(), 1):
                tweet_text = f"{i}. \"{row['tweet']}\" (Likes: {row['likes']}, Replies: {row['replies']}, Retweets: {row['retweets']})"
                pdf.chapter_body(clean_text(tweet_text))
    except Exception as e:
        pdf.chapter_body(f"Error processing tweet data: {str(e)}")
    
    # Generate PDF in memory using BytesIO
    pdf_output = io.BytesIO()
    # Use dest parameter with 'S' to specify it's a string output
    pdf_data = pdf.output(dest='S').encode('latin-1')
    
    # Put the data into the BytesIO object
    pdf_output.write(pdf_data)
    pdf_output.seek(0)
    
    # Convert to base64 for embedding in HTML
    b64_pdf = base64.b64encode(pdf_output.getvalue()).decode()
    
    return b64_pdf

# Streamlit app
def main():
    st.set_page_config(page_title="Twitter Data Analyzer", layout="wide", page_icon="🐦")
    
    st.title("🐦 Twitter Data Analyzer")
    st.markdown("Analyze tweet data with statistical and semantic analysis")
    
    # Initialize session state for conversation history
    if 'conversation_history' not in st.session_state:
        st.session_state['conversation_history'] = []
    
    if 'vectorstore_initialized' not in st.session_state:
        st.session_state['vectorstore_initialized'] = False
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # File upload
        st.header("Upload Data")
        uploaded_file = st.file_uploader("Upload your Twitter CSV file", type=["csv"])
        
        # Sample data option
        use_sample_data = st.checkbox("Use sample data instead", value=not uploaded_file)
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This app analyzes Twitter data, answering both statistical 
        and semantic questions about tweets using LangChain and Groq API.
        """)

    # Load data
    df = None
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded successfully")
    elif use_sample_data:
        # Create sample data
        df = pd.DataFrame({
            'fullname': ['Mahua Moitra', 'Mahua Moitra', 'Mahua Moitra'],
            'likes': [10897, 6377, 3873],
            'replies': [279, 453, 93],
            'retweets': [2909, 1623, 1049],
            'tweet': [
                "For those who truly want to understand why we are opposing the Waqf Act pls listen to @ShayarImran's brilliant speech in RS.",
                "Waqf act is discriminatory and seeks to create an entire class of second class citizens.",
                "We went to Election Commission @ECISVEEP to protest against duplicate EPIC issue & Adhar linkage to voter ID"
            ]
        })
        st.info("Using sample Twitter data")
    
    # If data is available, initialize analyzer and show interface
    if df is not None:
        # Validate required columns
        required_columns = ['fullname', 'likes', 'replies', 'retweets', 'tweet']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"❌ Missing required columns: {', '.join(missing_columns)}")
        else:
            # Initialize analyzer
            analyzer = TwitterAnalyzer(df)
            
            # Create tabs
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "❓ Question Answering", "📈 Visualization", "📄 Raw Data"])
            
            # Dashboard tab
            with tab1:
                st.header("Data Overview")
                
                # Get summary stats
                summary = analyzer.get_data_summary()
                
                # Create dashboard metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Tweets", summary["total_tweets"])
                with col2:
                    st.metric("Unique Users", summary["unique_users"])
                with col3:
                    st.metric("Avg. Likes", f"{summary['avg_likes']:.1f}")
                with col4:
                    st.metric("Avg. Retweets", f"{summary['avg_retweets']:.1f}")
                
                # Show schema
                st.subheader("Data Schema")
                st.code(analyzer.get_data_schema())
                
                # Most engaged tweets
                st.subheader("Most Engaged Tweets")
                
                max_likes_row = df.loc[df['likes'].idxmax()]
                max_retweets_row = df.loc[df['retweets'].idxmax()]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Most Liked")
                    st.markdown(f"**{max_likes_row['fullname']}**")
                    st.markdown(f"*{max_likes_row['tweet']}*")
                    st.metric("Likes", max_likes_row['likes'])
                
                with col2:
                    st.markdown("#### Most Retweeted")
                    st.markdown(f"**{max_retweets_row['fullname']}**")
                    st.markdown(f"*{max_retweets_row['tweet']}*")
                    st.metric("Retweets", max_retweets_row['retweets'])
                
                # User activity chart
                st.subheader("User Activity")
                user_tweet_counts = df['fullname'].value_counts().reset_index()
                user_tweet_counts.columns = ['User', 'Tweet Count']
                
                fig = px.bar(
                    user_tweet_counts, 
                    x='User', 
                    y='Tweet Count',
                    title="Tweets per User",
                    color='Tweet Count',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
                status_col1, status_col2 = st.columns(2)
                with status_col1:
                    if st.session_state['vectorstore_initialized']:
                        st.success("✅ Vector store initialized for semantic search")
                    else:
                        st.warning("⚠️ Vector store not initialized")
                
                with status_col2:
                    st.success("✅ News integration active with Gemini")

            # Question Answering Tab
            with tab2:
                st.header("Ask Questions About Your Data")
                
                # Display RAG initialization status
                if st.session_state['vectorstore_initialized']:
                    st.success("✅ Vector store initialized for semantic search")
                
                # Example questions
                st.markdown("### Example Questions:")
                example_questions = [
                    "What is the total sum of likes for Mahua Moitra?",
                    "Who has the highest number of likes?",
                    "What are the main themes in Mahua Moitra's tweets?",
                    "What does Mahua Moitra say about the Waqf Act?",
                    "What is the latest news about Mahua Moitra?",
                    "Why is Mahua Moitra tweeting about the Election Commission?",
                    "What is the context behind Mahua Moitra's Waqf Act tweet?",
                    "Compare Mahua Moitra's tweets with current news about her",
                    "Summarize how Mahua Moitra's tweets relate to recent political events"
                ]
                
                # Select example or enter custom question
                question_type = st.radio("Choose question type:", ["Select an example", "Ask your own question"])
                
                if question_type == "Select an example":
                    question = st.selectbox("Select a question:", example_questions)
                else:
                    question = st.text_input("Enter your question:")
                
                # Add a button to submit the question
                if st.button("Analyze") and question:
                    with st.spinner("Analyzing..."):
                        answer = analyzer.analyze(question)
                        
                        entities = analyzer.extract_entities(question)
                        
                        
                        st.markdown("### Answer:")
                        st.markdown(answer)
                        
                        # Extract entities using the existing function in the analyzer
                        # entities = analyzer.extract_entities(question)
                        # Step 3: Get and display news separately
                        with st.spinner("Checking latest news..."):
                            news_analysis = analyzer.get_news_analysis(entities,question)
                            if news_analysis:
                                st.markdown("### Latest Related News:")
                                st.markdown(news_analysis)
                        
                        # Add to conversation history
                        st.session_state['conversation_history'].append({"question": question, "answer": answer,"entities": entities,"news": news_analysis})

                        if entities:
                            # Use the first entity as the main person for the report
                            person_name = entities[0]
                            
                            st.markdown("---")
                            st.markdown(f"### Generate PDF Report for {person_name}")
                            
                            # Generate PDF
                            pdf_b64 = generate_person_pdf(analyzer, person_name, question, answer,news_analysis)
                            
                            # Create download link
                            # download_link = create_download_link(pdf_b64, f"{person_name}_analysis.pdf")
                            
                            # Preview and download section
                            st.markdown("**PDF Preview**")
                            st.markdown(f"""
                            <iframe
                                title="PDF Preview"
                                src="data:application/pdf;base64,{pdf_b64}"
                                width="100%"
                                height="500px"
                                style="border: none;"
                            ></iframe>
                            """, unsafe_allow_html=True)
                            
                
                # Show conversation history
                if st.session_state['conversation_history']:
                    st.markdown("### Recent Questions")
                    for i, qa in enumerate(reversed(st.session_state['conversation_history'][-5:])):
                        with st.expander(f"Q: {qa['question']}"):
                            st.markdown(qa['answer'])
            
            # Visualization Tab
            with tab3:
                st.header("Data Visualization")
                
                # Visualization options
                viz_type = st.radio("Select visualization type:", ["User Statistics", "User Comparison"])
                
                if viz_type == "User Statistics":
                    # User selection or overall
                    all_users = ["All Users"] + list(df['fullname'].unique())
                    selected_user = st.selectbox("Select user:", all_users)
                    
                    if selected_user == "All Users":
                        fig, error = analyzer.generate_user_statistics_chart()
                    else:
                        fig, error = analyzer.generate_user_statistics_chart(selected_user)
                    
                    if error:
                        st.error(error)
                    elif fig:
                        st.plotly_chart(fig, use_container_width=True)
                
                elif viz_type == "User Comparison":
                    # Select two users to compare
                    users = list(df['fullname'].unique())
                    if len(users) < 2:
                        st.warning("Need at least two users in the dataset for comparison")
                    else:
                        col1, col2 = st.columns(2)
                        with col1:
                            user1 = st.selectbox("Select first user:", users)
                        with col2:
                            # Filter out first user from second dropdown
                            remaining_users = [u for u in users if u != user1]
                            user2 = st.selectbox("Select second user:", remaining_users if remaining_users else users)
                        
                        fig, error = analyzer.generate_comparison_chart(user1, user2)
                        
                        if error:
                            st.error(error)
                        elif fig:
                            st.plotly_chart(fig, use_container_width=True)
            
            # Raw Data Tab
            with tab4:
                st.header("Raw Data")
                st.dataframe(df)
    else:
        st.info("Please upload a CSV file or use the sample data to start analyzing")

if __name__ == "__main__":
    main()
