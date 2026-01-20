*🌍 Language Detection using Recurrent Neural Network (RNN)*



*This project predicts the language of a given text sentence using a Recurrent Neural Network (RNN) trained on multilingual text data.*

*It demonstrates a complete end-to-end machine learning workflow, including data analysis, model training, prediction, and deployment using Streamlit.*



*📌 Project Description*



*Language detection is a fundamental Natural Language Processing (NLP) task used in applications such as:*



*Multilingual chat systems*



*Search engines*



*Translation pipelines*



*In this project, a Simple RNN model is trained to classify text into its corresponding language based on learned textual patterns.*



*The project is divided into three major components:*



*Exploratory Data Analysis \& Preprocessing*



*Model Training \& Prediction Pipeline*



*Streamlit Web Application*



*🧠 1. Exploratory Data Analysis \& Preprocessing*



*Performed analysis on multilingual text data*



*Cleaned and prepared text for sequence modeling*



*Converted text into numerical sequences using a tokenizer*



*Encoded language labels using label encoding*



*Relevant file:*



*eda.ipynb* 



*app*



*🧠 2. Model Building \& Training*



*Built a Simple RNN-based neural network using TensorFlow \& Keras*



*Used padded sequences for uniform input length*



*Trained the model to learn sequential language patterns*



*Saved trained artifacts for reuse in deployment:*



*Trained RNN model (simple\_rnn\_model.h5)*



*Tokenizer \& label encoder (tokenizer.pkl)*



*Relevant file:*



*prediction.ipynb* 



*app*



*🔮 3. Prediction Pipeline*



*The prediction pipeline:*



*Loads the trained RNN model*



*Loads the tokenizer and label encoder*



*Converts user input text into padded sequences*



*Predicts the language and confidence score*



*This ensures consistent preprocessing between training and inference.*



*🌐 4. Streamlit Web Application*



*An interactive Streamlit app that allows users to:*



*Enter any sentence*



*Instantly detect the predicted language*



*View the model’s confidence score*



*Key features:*



*Clean UI*



*Cached model loading for efficiency*



*Real-time inference*



*File:*



*app.py* 



*app*



*🧾 Input*



*Any text sentence (single or multiple words)*



*Example:*



*यह एक अच्छा दिन है*



*📤 Output*



*Predicted Language*



*Confidence Score*



*🛠️ Tech Stack Used*

*Programming \& ML*



*Python*



*TensorFlow*



*Keras*



*NLP \& Data Processing*



*NumPy*



*Pandas*



*Scikit-learn*



*Visualization \& Analysis*



*Matplotlib*



*Seaborn*



*Deployment*



*Streamlit*



*Utilities*



*Pickle*



*IPyKernel*



*📦 Installation \& Setup*

*Step 1: Clone the Repository*

*git clone https://github.com/ArnabKrJana/language-detection-rnn.git*

*cd language-detection-rnn*



*Step 2: Install Dependencies*

*pip install -r requirements.txt*



*Step 3: Run the Streamlit App*

*streamlit run app.py*



*📂 Project Structure*

*├── app.py*

*├── eda.ipynb*

*├── prediction.ipynb*

*├── requirements.txt*

*├── saved\_model/*

*│   ├── simple\_rnn\_model.h5*

*│   └── tokenizer.pkl*



*👤 Author Details*



*Name: Arnab Kumar Jana*



*LinkedIn:*

*https://www.linkedin.com/in/arnab-kumar-jana-827420377*



*GitHub:*

*https://github.com/ArnabKrJana*



*✅ Future Improvements*



*Support for more languages*



*Use LSTM / GRU for better sequence learning*



*Add confusion matrix \& evaluation metrics*



*Deploy on Streamlit Cloud*

