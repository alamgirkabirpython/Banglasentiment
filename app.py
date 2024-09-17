
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout, LSTM, Bidirectional
import streamlit as st
import pandas as pd
import json
import bcrypt
import nltk
# Set a download path in the temporary directory
nltk_data_dir = '/tmp/nltk_data'

# Ensure the directory exists
os.makedirs(nltk_data_dir, exist_ok=True)

# Download the 'punkt' tokenizer to the local directory
nltk.download('punkt', download_dir=nltk_data_dir)

# Append the local directory to NLTK's data path
nltk.data.path.append(nltk_data_dir)

def model_train(df, text_column, label_column):
    encoder = LabelEncoder()
    df[label_column] = encoder.fit_transform(df[label_column])
    
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(df[text_column])
    sequences = tokenizer.texts_to_sequences(df[text_column])
    max_length = max([len(x) for x in sequences])
    
    X = pad_sequences(sequences, maxlen=max_length, padding='post')
    y = df[label_column]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    vocab_size = len(tokenizer.word_index) + 1
    embedding_dim = 64

    # Model 1: Conv1D-based model
    model_conv1d = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
        Conv1D(128, 5, activation='relu'),
        GlobalMaxPooling1D(),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dense(len(y.unique()), activation='softmax')
    ])

    model_conv1d.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model_conv1d.fit(X_train, y_train, epochs=15, validation_data=(X_test, y_test), batch_size=32)

    # Model 2: Bidirectional LSTM-based model
    model_lstm = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
        Bidirectional(LSTM(64, return_sequences=True)),
        GlobalMaxPooling1D(),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dense(len(y.unique()), activation='softmax')
    ])

    model_lstm.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model_lstm.fit(X_train, y_train, epochs=15, validation_data=(X_test, y_test), batch_size=32)
    
    return model_conv1d, model_lstm, tokenizer, encoder, X_test, y_test, max_length


def predict_sentiment(text, model, tokenizer, encoder,max_length):
    text_seq = tokenizer.texts_to_sequences([text])
    text_pad = pad_sequences(text_seq, maxlen=max_length, padding='post')
    prediction = model.predict(text_pad)
    predicted_label = encoder.inverse_transform([prediction.argmax(axis=1)[0]])[0]
    return predicted_label


import nltk
import re

nltk.download('punkt')

emoji_dict = {
    "😊": "হাসি",
    "😢": "কান্না",
    "😠": "রাগ",
    "😔": "মন খারাপ",
    "👍": "দারুণ",
    "😎": "ঠান্ডা",
    "😭": "অশ্রু",
    "😁": "মুচকি হাসি",
    "😅": "হালকা হাসি",
    "😍": "ভালবাসা",
    "😒": "অসন্তুষ্ট",
    "😞": "হতাশা",
    "😡": "রাগান্বিত",
    "😃": "খুশি",
    "😉": "চোখ মারা",
    "😋": "স্বাদ আস্বাদন",
    "😐": "নির্বিকার",
    "😤": "অসন্তুষ্ট",
    "😴": "ঘুম",
    "😜": "মজার",
    "😩": "ক্লান্ত",
    "😯": "আশ্চর্য",
    "😆": "হাসি",
    "😷": "মাস্ক",
    "🙄": "গুর্গুরানি",
    "😳": "বিস্মিত",
    "😬": "চিন্তিত",
    "😚": "চুম্বন",
    "😰": "উদ্বিগ্ন",
    "🤗": "আলিঙ্গন",
    "🤔": "চিন্তাশীল",
    "🤐": "মুখ বন্ধ",
    "😇": "পুণ্যবান"
}

bn_stopwords = ['আমি', 'আপনি', 'সে', 'তুমি', 'আমরা', 'আপনারা', 'তারা', 'এর', 'এই', 'ওই', 'তার', 'তাদের', 'আমাদের', 'আপনার', 'তোমার', 'আপনারা', 'আমরা', 'যে', 'যত', 'সব', 'কিছু', 'বহু', 'কোনো', 'কোন', 'একটি', 'এটি', 'তিন', 'চার', 'এখানে', 'যেখানে', 'এখানে', 'কিভাবে', 'কীভাবে', 'কেন', 'কেননা', 'যখন', 'যদি', 'তবে', 'কিন্তু', 'তবে', 'আর', 'তিন', 'চার', 'অথবা', 'নাহলে', 'যত', 'যেমন', 'কিন্তু', 'নিশ্চিত', 'সবাই', 'অনেক', 'কিছু', 'কেননা', 'অন্য', 'নতুন', 'পুরানো', 'বেশি', 'কম', 'অন্তত', 'বেশি', 'কম', 'এবং', 'অথবা', 'নির্বাচন', 'এই', 'ওই', 'ফিরে', 'তারপর', 'পরে', 'আগে', 'তখন', 'নতুন', 'পুরানো', 'যে', 'তবে', 'অথবা', 'একটি', 'অথবা', 'অবশ্য', 'এরপর', 'আমরা', 'বিভিন্ন', 'সকল', 'যেখানে', 'এখানে', 'কি', 'মাঝে', 'মধ্যে', 'মধ্যবর্তী', 'যে', 'শুধু', 'উল্লেখযোগ্য', 'অধিক', 'যেমন', 'বিভিন্ন', 'অপর', 'অন্য', 'কিছু', 'অন্যান্য', 'আর', 'যাওয়া', 'আসা', 'কী', 'যখন', 'এটি', 'কারণে', 'তারপর', 'তাদের', 'আমাদের', 'আরও', 'অবশ্যই', 'এবং', 'অথবা', 'বিশেষ', 'কি', 'ব্যাপারে', 'অথবা', 'দ্বারা', 'তারা', 'এক', 'মধ্যে', 'কিছু', 'তবে', 'এছাড়া', 'মধ্যে', 'কোনো', 'অন্য', 'প্রতিটি', 'একটি', 'যেখানে', 'যত', 'যে', 'যদি', 'আর', 'অনেক', 'যেমন', 'যেমন', 'তার', 'ভেতরে', 'দ্বারা', 'এর', 'আছে', 'দিয়ে', 'যাওয়া', 'আসা', 'যে', 'নতুন', 'পুরানো', 'যেমন', 'ফিরে', 'পরের', 'সকল', 'তাদের', 'সব', 'অন্যান্য', 'আরও', 'কোন', 'এখানে', 'যখন', 'তবে', 'তাদের', 'ফিরে', 'যেখানে', 'আরও', 'আমরা', 'কিছু', 'অন্য', 'নতুন', 'কিছু', 'অন্য', 'সবার', 'অপর', 'মাঝে', 'বিভিন্ন', 'এই', 'তাদের', 'আমাদের', 'এই', 'যেমন', 'অতএব', 'এরপর', 'নতুন', 'সর্বোচ্চ', 'সর্বনিম্ন', 'মাঝে', 'আমি', 'আপনি', 'সে', 'তুমি', 'আমরা', 'আপনারা', 'তারা', 'এর', 'এই', 'ওই', 'তার', 'তাদের', 'আমাদের', 'আপনার', 'তোমার', 'আপনারা', 'আমরা', 'যে', 'যত', 'সব', 'কিছু', 'বহু', 'কোনো', 'কোন', 'একটি', 'এটি', 'তিন', 'চার', 'এখানে', 'যেখানে', 'এখানে', 'কিভাবে', 'কীভাবে', 'কেন', 'কেননা', 'যখন', 'যদি', 'তবে', 'কিন্তু', 'তবে', 'আর', 'তিন', 'চার', 'অথবা', 'নাহলে', 'যত', 'যেমন', 'কিন্তু', 'নিশ্চিত', 'সবাই', 'অনেক', 'কিছু', 'কেননা', 'অন্য', 'নতুন', 'পুরানো', 'বেশি', 'কম', 'অন্তত', 'বেশি', 'কম', 'এবং', 'অথবা', 'নির্বাচন', 'এই', 'ওই', 'ফিরে', 'তারপর', 'পরে', 'আগে', 'তখন', 'নতুন', 'পুরানো', 'যে', 'তবে', 'অথবা', 'একটি', 'অথবা', 'অবশ্য', 'এরপর', 'আমরা', 'বিভিন্ন', 'সকল', 'যেখানে', 'এখানে', 'কি', 'মাঝে', 'মধ্যে', 'মধ্যবর্তী', 'যে', 'শুধু', 'উল্লেখযোগ্য', 'অধিক', 'যেমন', 'বিভিন্ন', 'অপর', 'অন্য', 'কিছু', 'অন্যান্য', 'আর', 'যাওয়া', 'আসা', 'কী', 'যখন', 'এটি', 'কারণে', 'তারপর', 'তাদের', 'আমাদের', 'আরও', 'অবশ্যই', 'এবং', 'অথবা', 'বিশেষ', 'কি', 'ব্যাপারে', 'অথবা', 'দ্বারা', 'তারা', 'এক', 'মধ্যে', 'কিছু', 'তবে', 'এছাড়া', 'মধ্যে', 'কোনো', 'অন্য', 'প্রতিটি', 'একটি', 'যেখানে', 'যত', 'যে', 'যদি', 'আর', 'অনেক', 'যেমন', 'যেমন', 'তার', 'ভেতরে', 'দ্বারা', 'এর', 'আছে', 'দিয়ে', 'যাওয়া', 'আসা', 'যে', 'নতুন', 'পুরানো', 'যেমন', 'ফিরে', 'পরের', 'সকল', 'তাদের', 'সব', 'অন্যান্য', 'আরও', 'কোন', 'এখানে', 'যখন', 'তবে', 'তাদের', 'ফিরে', 'যেখানে', 'আরও', 'আমরা', 'কিছু', 'অন্য', 'নতুন', 'কিছু', 'অন্য', 'সবার', 'অপর', 'মাঝে', 'বিভিন্ন', 'এই', 'তাদের', 'আমাদের', 'এই', 'যেমন', 'অতএব', 'এরপর', 'নতুন', 'সর্বোচ্চ', 'সর্বনিম্ন', 'মাঝে', 'আমি', 'আপনি', 'সে', 'তুমি', 'আমরা', 'আপনারা', 'তারা', 'এর', 'এই', 'ওই', 'তার', 'তাদের', 'আমাদের', 'আপনার', 'তোমার', 'আপনারা', 'আমরা', 'যে', 'যত', 'সব', 'কিছু', 'বহু', 'কোনো', 'কোন', 'একটি', 'এটি', 'তিন', 'চার', 'এখানে', 'যেখানে', 'এখানে', 'কিভাবে', 'কীভাবে', 'কেন', 'কেননা', 'যখন', 'যদি', 'তবে', 'কিন্তু', 'তবে', 'আর', 'তিন', 'চার', 'অথবা', 'নাহলে', 'যত', 'যেমন', 'কিন্তু', 'নিশ্চিত', 'সবাই', 'অনেক', 'কিছু', 'কেননা', 'অন্য', 'নতুন', 'পুরানো', 'বেশি', 'কম', 'অন্তত', 'বেশি', 'কম', 'এবং', 'অথবা', 'নির্বাচন', 'এই', 'ওই', 'ফিরে', 'তারপর', 'পরে', 'আগে', 'তখন', 'নতুন', 'পুরানো', 'যে', 'তবে', 'অথবা', 'একটি', 'অথবা', 'অবশ্য', 'এরপর', 'আমরা', 'বিভিন্ন', 'সকল', 'যেখানে', 'এখানে', 'কি', 'মাঝে', 'মধ্যে', 'মধ্যবর্তী', 'যে', 'শুধু', 'উল্লেখযোগ্য', 'অধিক', 'যেমন', 'বিভিন্ন', 'অপর', 'অন্য', 'কিছু', 'অন্যান্য', 'আর', 'যাওয়া', 'আসা', 'কী', 'যখন', 'এটি', 'কারণে', 'তারপর', 'তাদের', 'আমাদের', 'আরও', 'অবশ্যই', 'এবং', 'অথবা', 'বিশেষ', 'কি', 'ব্যাপারে', 'অথবা', 'দ্বারা', 'তারা', 'এক', 'মধ্যে', 'কিছু', 'তবে', 'এছাড়া', 'মধ্যে', 'কোনো', 'অন্য', 'প্রতিটি', 'একটি', 'যেখানে', 'যত', 'যে', 'যদি', 'আর', 'অনেক', 'যেমন', 'যেমন', 'তার', 'ভেতরে', 'দ্বারা', 'এর', 'আছে', 'দিয়ে', 'যাওয়া', 'আসা', 'যে', 'নতুন', 'পুরানো', 'যেমন', 'ফিরে', 'পরের', 'সকল', 'তাদের', 'সব', 'অন্যান্য', 'আরও', 'কোন', 'এখানে', 'যখন', 'তবে', 'তাদের', 'ফিরে', 'যেখানে', 'আরও', 'আমরা', 'কিছু', 'অন্য', 'নতুন', 'কিছু', 'অন্য', 'সবার', 'অপর', 'মাঝে', 'বিভিন্ন', 'এই', 'তাদের', 'আমাদের', 'এই', 'যেমন', 'অতএব', 'এরপর', 'নতুন', 'সর্বোচ্চ', 'সর্বনিম্ন', 'মাঝে', 'অতএব', 'অথচ', 'অথবা', 'অনুযায়ী', 'অনেক', 'অনেকে', 'অনেকেই', 'অন্তত', 'অন্য', 'অবধি', 'অবশ্য', 'অর্থাত', 'আই', 'আগামী', 'আগে', 'আগেই', 'আছে', 'আজ', 'আদ্যভাগে', 'আপনার', 'আপনি', 'আবার', 'আমরা', 'আমাকে', 'আমাদের', 'আমার', 'আমি', 'আর', 'আরও', 'ই', 'ইত্যাদি', 'ইহা', 'উচিত', 'উত্তর', 'উনি', 'উপর', 'উপরে', 'এ', 'এঁদের', 'এঁরা', 'এই', 'একই', 'একটি', 'একবার', 'একে', 'এক্', 'এখন', 'এখনও', 'এখানে', 'এখানেই', 'এটা', 'এটাই', 'এটি', 'এত', 'এতটাই', 'এতে', 'এদের', 'এব', 'এবং', 'এবার', 'এমন', 'এমনকী', 'এমনি', 'এর', 'এরা', 'এল', 'এস', 'এসে', 'ঐ', 'ও', 'ওঁদের', 'ওঁর', 'ওঁরা', 'ওই', 'ওকে', 'ওখানে', 'ওদের', 'ওর', 'ওরা', 'কখনও', 'কত', 'কবে', 'কমনে', 'কয়েক', 'কয়েকটি', 'করছে', 'করছেন', 'করতে', 'করবে', 'করবেন', 'করলে', 'করলেন', 'করা', 'করাই', 'করায়', 'করার', 'করি', 'করিতে', 'করিয়া', 'করিয়ে', 'করে', 'করেই', 'করেছিলেন', 'করেছে', 'করেছেন', 'করেন', 'কাউকে', 'কাছ', 'কাছে', 'কাজ', 'কাজে', 'কারও', 'কারণ', 'কি', 'কিংবা', 'কিছু', 'কিছুই', 'কিন্তু', 'কী', 'কে', 'কেউ', 'কেউই', 'কেখা', 'কেন', 'কোটি', 'কোন', 'কোনও', 'কোনো', 'ক্ষেত্রে', 'কয়েক', 'খুব', 'গিয়ে', 'গিয়েছে', 'গিয়ে', 'গুলি', 'গেছে', 'গেল', 'গেলে', 'গোটা', 'চলে', 'চান', 'চায়', 'চার', 'চালু', 'চেয়ে', 'চেষ্টা', 'ছাড়া', 'ছাড়াও', 'ছিল', 'ছিলেন', 'জন', 'জনকে', 'জনের', 'জন্য', 'জন্যওজে', 'জানতে', 'জানা', 'জানানো', 'জানায়', 'জানিয়ে', 'জানিয়েছে', 'জে', 'জ্নজন', 'টি', 'ঠিক', 'তখন', 'তত', 'তথা', 'তবু', 'তবে', 'তা', 'তাঁকে', 'তাঁদের', 'তাঁর', 'তাঁরা', 'তাঁাহারা', 'তাই', 'তাও', 'তাকে', 'তাতে', 'তাদের', 'তার', 'তারপর', 'তারা', 'তারৈ', 'তাহলে', 'তাহা', 'তাহাতে', 'তাহার', 'তিনঐ', 'তিনি', 'তিনিও', 'তুমি', 'তুলে', 'তেমন', 'তো', 'তোমার', 'থাকবে', 'থাকবেন', 'থাকা', 'থাকায়', 'থাকে', 'থাকেন', 'থেকে', 'থেকেই', 'থেকেও', 'দিকে', 'দিতে', 'দিন', 'দিয়ে', 'দিয়েছে', 'দিয়েছেন', 'দিলেন', 'দু', 'দুই', 'দুটি', 'দুটো', 'দেওয়া', 'দেওয়ার', 'দেওয়া', 'দেখতে', 'দেখা', 'দেখে', 'দেন', 'দেয়', 'দ্বারা', 'ধরা', 'ধরে', 'ধামার', 'নতুন', 'নয়', 'না', 'নাই', 'নাকি', 'নাগাদ', 'নানা', 'নিজে', 'নিজেই', 'নিজেদের', 'নিজের', 'নিতে', 'নিয়ে', 'নিয়ে', 'নেই', 'নেওয়া', 'নেওয়ার', 'নেওয়া', 'নয়', 'পক্ষে', 'পর', 'পরে', 'পরেই', 'পরেও', 'পর্যন্ত', 'পাওয়া', 'পাচ', 'পারি', 'পারে', 'পারেন', 'পি', 'পেয়ে', 'পেয়্র্', 'প্রতি', 'প্রথম', 'প্রভৃতি', 'প্রযন্ত', 'প্রাথমিক', 'প্রায়', 'প্রায়', 'ফলে', 'ফিরে', 'ফের', 'বক্তব্য', 'বদলে', 'বন', 'বরং', 'বলতে', 'বলল', 'বললেন', 'বলা', 'বলে', 'বলেছেন', 'বলেন', 'বসে', 'বহু', 'বা', 'বাদে', 'বার', 'বি', 'বিনা', 'বিভিন্ন', 'বিশেষ', 'বিষয়টি', 'বেশ', 'বেশি', 'ব্যবহার', 'ব্যাপারে', 'ভাবে', 'ভাবেই', 'মতো', 'মতোই', 'মধ্যভাগে', 'মধ্যে', 'মধ্যেই', 'মধ্যেও', 'মনে', 'মাত্র', 'মাধ্যমে', 'মোট', 'মোটেই', 'যখন', 'যত', 'যতটা', 'যথেষ্ট', 'যদি', 'যদিও', 'যা', 'যাঁর', 'যাঁরা', 'যাওয়া', 'যাওয়ার', 'যাওয়া', 'যাকে', 'যাচ্ছে', 'যাতে', 'যাদের', 'যান', 'যাবে', 'যায়', 'যার', 'যারা', 'যিনি', 'যে', 'যেখানে', 'যেতে', 'যেন', 'যেমন', 'র', 'রকম', 'রয়েছে', 'রাখা', 'রেখে', 'লক্ষ', 'শুধু', 'শুরু', 'সঙ্গে', 'সঙ্গেও', 'সব', 'সবার', 'সমস্ত', 'সম্প্রতি', 'সহ', 'সহিত', 'সাধারণ', 'সামনে', 'সি', 'সুতরাং', 'সে', 'সেই', 'সেখান', 'সেখানে', 'সেটা', 'সেটাই', 'সেটাও', 'সেটি', 'স্পষ্ট', 'স্বয়ং', 'হইতে', 'হইবে', 'হইয়া', 'হওয়া', 'হওয়ায়', 'হওয়ার', 'হচ্ছে', 'হত', 'হতে', 'হতেই', 'হন', 'হবে', 'হবেন', 'হয়', 'হয়তো', 'হয়নি', 'হয়ে', 'হয়েই', 'হয়েছিল', 'হয়েছে', 'হয়েছেন', 'হল', 'হলে', 'হলেই', 'হলেও', 'হলো', 'হাজার', 'হিসাবে', 'হৈলে', 'হোক', 'হয়']

def remove_stopwords(text):
    words = nltk.word_tokenize(text)
    filtered_words = [word for word in words if word not in bn_stopwords]
    return ' '.join(filtered_words)

def remove_non_bangla_words(text):
    return ' '.join([word for word in text.split() if all('ঀ' <= c <= '৿' for c in word)])

def remove_null_characters(text):
    return text.replace('\x00', '')

def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def replace_emojis(text):
    for emoji, meaning in emoji_dict.items():
        text = text.replace(emoji, meaning)
    return text

def preprocess_text(text):
    text = replace_emojis(text)
    text = remove_stopwords(text)
    text = remove_non_bangla_words(text)
    text = remove_null_characters(text)
    text = remove_urls(text)
    return text


# App logo and background images
logo_url = "https://img.freepik.com/free-vector/colorful-bird-illustration-gradient_343694-1741.jpg?size=626&ext=jpg&ga=GA1.1.733875022.1726100029&semt=ais_hybrid.png"
background_image_url = "https://images.unsplash.com/photo-1507525428034-b723cf961d3e?crop=entropy&cs=tinysrgb&w=1080&fit=max"

# Set up Streamlit app configuration
st.set_page_config(page_title="Bangla Sentiment Analysis", page_icon=logo_url, layout="wide")

# Custom CSS for styling the app with background image
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("{background_image_url}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    .stButton>button {{
        background-color: #3498db;
        color: white;
        border-radius: 10px;
    }}
    .stButton>button:hover {{
        background-color: #2980b9;
    }}
    .stTextInput, .stTextArea {{
        background-color: rgba(255, 255, 255, 0.8);
    }}
    h1, h2, h3 {{
        color: white;
        text-align: center;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
    }}
    .centered {{
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100%;
    }}
    .dataframe-container {{
        width: 80%;
        margin: 0 auto;
        height: 150px;
        overflow-y: auto;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Load users from a JSON file
users_file = 'users.json'

def load_users():
    if os.path.exists(users_file):
        with open(users_file, 'r') as file:
            return json.load(file)
    else:
        return {}

# Save users to a JSON file
def save_users(users_db):
    with open(users_file, 'w') as file:
        json.dump(users_db, file)

# Load user database from file
users_db = load_users()

# Caching the data loading function
@st.cache_data
def data_load(file_path):
    df = pd.read_csv(file_path)
    if df.isnull().values.any():
        st.warning("Warning: The dataset contains missing values. Please clean the data.")
    df['processed_text'] = df['text'].apply(preprocess_text)
    return df

# Caching the model training function
@st.cache_resource
def train_model(file_path):
    df = data_load(file_path)
    model1, model2, tokenizer, encoder, X_test, y_test, max_length = model_train(df, 'processed_text', 'label')
    return model1, model2, tokenizer, encoder, X_test, y_test, max_length

# Load the dataset and train models
file_path = "https://raw.githubusercontent.com/alamgirkabirpython/Banglasentiment/17d631b16b6e920ed8a2c8057a544ebb4e5d81db/bangla_sentiment_data.csv"
model1, model2, tokenizer, encoder, X_test, y_test, max_length = train_model(file_path)

# Function to hash passwords before storing them
def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

# Function to verify hashed passwords
def check_password(password, hashed):
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

# Display the app logo
st.image(logo_url, width=100)

# Main page title
st.title("Bangla Sentiment Analysis :heart_eyes: :cry:")

# Sidebar with example input texts
with st.sidebar:
    st.write("### Input Text Example:")
    st.code("আমি খুব খারাপ আছি 😢", language="plain")
    st.code("তুমি কেন এমন করলে? 😡", language="plain")
    st.code("তুমি আমাকে রাগিয়ে দিচ্ছ 😠", language="plain")
    st.code("আজকে সবকিছুই বিরক্তিকর 😡", language="plain")
    st.code("আজ আমার খুব মন খারাপ।", language="plain")
    st.code("আমি খুব একা বোধ করছি।", language="plain")
    st.code("জীবনটা কেন এত কঠিন!", language="plain")
    st.code("কিছুই যেন আর ভালো লাগছে না।", language="plain")
    st.code("মনে হচ্ছে সব কিছু ভেঙে পড়ছে।", language="plain")
    st.code("আজ আমি খুব আনন্দিত!", language="plain")
    st.code("এটা আমার জীবনের সেরা মুহূর্ত!", language="plain")
    st.code("সবকিছু এত সুন্দর লাগছে!", language="plain")
    st.code("আজকের দিনটা সত্যিই অসাধারণ!", language="plain")

# Create navigation buttons
col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
with col1:
    login_button = st.button("Login", key="login_button")
with col2:
    signup_button = st.button("Sign Up", key="signup_button")
with col3:
    data_button = st.button("Data", key="data_button")
with col4:
    contact_button = st.button("Contact Information", key="contact_button")

# Manage session state for navigation
if "selected_option" not in st.session_state:
    st.session_state["selected_option"] = None

if login_button:
    st.session_state["selected_option"] = "Login"
elif signup_button:
    st.session_state["selected_option"] = "Sign Up"
elif data_button:
    st.session_state["selected_option"] = "Data"
elif contact_button:
    st.session_state["selected_option"] = "Contact Information"

# Handle user authentication (Login and Signup)
if st.session_state["selected_option"] == "Login":
    st.subheader("Login")
    login_username = st.text_input("Username", key="login_username_input")
    login_password = st.text_input("Password", type="password", key="login_password_input")
    if st.button("Login", key="login_confirm_button"):
        if login_username in users_db and check_password(login_password, users_db[login_username]):
            st.session_state["authenticated"] = True
            st.session_state["username"] = login_username
            st.success(f"Welcome {login_username}! You are now logged in.")
        else:
            st.error("Invalid username or password.")

elif st.session_state["selected_option"] == "Sign Up":
    st.subheader("Sign Up")
    new_username = st.text_input("Choose a Username", key="signup_username_input")
    new_password = st.text_input("Choose a Password", type="password", key="signup_password_input")
    if st.button("Sign Up", key="signup_confirm_button"):
        if new_username in users_db:
            st.warning("Username already exists. Please choose a different username.")
        elif new_username and new_password:
            users_db[new_username] = hash_password(new_password)
            save_users(users_db)
            st.success("Sign up successful! You can now log in.")
        else:
            st.error("Please fill both fields.")

# Handle data and model display
elif st.session_state["selected_option"] == "Data":
    st.subheader("Data")
    if "authenticated" in st.session_state and st.session_state["authenticated"]:
        df = data_load(file_path)

        # Center the DataFrame
        st.markdown(
            """
            <div class="centered">
                <div class="dataframe-container">
                    {dataframe}
                </div>
            </div>
            """.format(dataframe=df.to_html(index=False, escape=False)),
            unsafe_allow_html=True
        )

        # Show model performance
        st.write("### Model Performance")
        model_choice = st.selectbox("Choose a model:", ['Model 1', 'Model 2'])
        model = model1 if model_choice == 'Model 1' else model2

        if model:
            loss, accuracy = model.evaluate(X_test, y_test)
            st.metric(label="Accuracy", value=f"{accuracy:.2f}")
            st.progress(accuracy)

        # Sentiment prediction
        st.write("## Predict Sentiment")
        user_input = st.text_area("Enter Bangla text for prediction", "")
        if st.button("Show Prediction", key="predict_button"):
            if user_input:
                predicted_label = predict_sentiment(user_input, model, tokenizer, encoder, max_length)
                st.success(f"Predicted Sentiment: {predicted_label}")
            else:
                st.warning("Please enter some text for prediction.")

elif st.session_state["selected_option"] == "Contact Information":
    st.subheader("Contact Information")
    st.write("Email: alomgirkabir720@gmail.com")
    st.write("Phone: +880-1234567890")
