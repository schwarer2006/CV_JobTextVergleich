import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from wordcloud import WordCloud
import PyPDF2
import pandas as pd
from datetime import date
from fpdf import FPDF
import json
import os
import io
import tempfile


# Funktion zur Textbereinigung
def clean_text(text):
    stopwords = ["von", "und", "bei", "mit", "für", "zur", "der", "September", "zum", "den"]
    for word in stopwords:
        text = text.replace(f" {word} ", " ")
    return text.lower()


# Funktion zum Extrahieren von Text aus einer PDF-Datei
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        text += pdf_reader.pages[page_num].extract_text()
    return text


# Funktion zum Erstellen einer Wordcloud
def create_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    return wordcloud


# Funktion zum Erstellen eines Radarplots
def create_radar_chart(labels, values1, values2):
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    values1 = values1.tolist()
    values2 = values2.tolist()

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, values1, color='blue', alpha=0.25)
    ax.fill(angles, values2, color='red', alpha=0.25)

    ax.set_yticklabels([])
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, rotation=90)

    return fig


# Funktion zum Speichern von Matplotlib-Figuren als temporäre Datei
def save_fig_to_tempfile(fig):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    fig.savefig(temp_file.name)
    temp_file.close()
    return temp_file.name


# Funktion zum Erstellen einer PDF
def create_pdf(name, lastname, job_title, job_description, similarity_score, figures):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Name: {name} {lastname}", ln=True)
    pdf.cell(200, 10, txt=f"Beruf: {job_title}", ln=True)
    pdf.cell(200, 10, txt=f"Stellenbeschreibung: {job_description}", ln=True)
    pdf.cell(200, 10, txt=f"Tagesdatum: {date.today().strftime('%d.%m.%Y')}", ln=True)
    pdf.cell(200, 10, txt=f"Ähnlichkeits-Score: {similarity_score:.2f}", ln=True)

    for title, fig_path in figures.items():
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(200, 10, txt=title, ln=True, align='C')
        pdf.image(fig_path, x=10, y=30, w=180)

    return pdf.output(dest="S").encode("latin1")


# Funktion zum Laden von gespeicherten Daten
def load_saved_data():
    if os.path.exists('saved_data.json'):
        with open('saved_data.json', 'r') as file:
            return json.load(file)
    return {}


# Funktion zum Speichern von Daten
def save_data(data):
    with open('saved_data.json', 'w') as file:
        json.dump(data, file)


# Initialisiere Session State und lade gespeicherte Daten
saved_data = load_saved_data()
if 'cv_text' not in st.session_state:
    st.session_state.cv_text = saved_data.get('cv_text', "")
if 'job_text' not in st.session_state:
    st.session_state.job_text = saved_data.get('job_text', "")
if 'cv_file' not in st.session_state:
    st.session_state.cv_file = saved_data.get('cv_file', None)
if 'job_file' not in st.session_state:
    st.session_state.job_file = saved_data.get('job_file', None)

# Persönliche Informationen laden
name = st.text_input("Name", value=saved_data.get('name', ""))
lastname = st.text_input("Nachname", value=saved_data.get('lastname', ""))
job_title = st.text_input("Beruf", value=saved_data.get('job_title', ""))
job_description = st.text_input("Stellenbeschreibung", value=saved_data.get('job_description', ""))

# Datei-Uploads
uploaded_cv_file = st.file_uploader("Lebenslauf hochladen (PDF oder Text)", type=["pdf", "txt"])
uploaded_job_file = st.file_uploader("Stellenbeschreibung hochladen (PDF oder Text)", type=["pdf", "txt"])

# Texteingabe direkt
cv_text_input = st.text_area("Lebenslauf eingeben", height=200, value=saved_data.get('cv_text_input', ""))
job_text_input = st.text_area("Stellenbeschreibung eingeben", height=200, value=saved_data.get('job_text_input', ""))

if st.button("Berechne Ähnlichkeit"):
    if uploaded_cv_file:
        if uploaded_cv_file.type == "application/pdf":
            st.session_state.cv_text = extract_text_from_pdf(uploaded_cv_file)
        elif uploaded_cv_file.type == "text/plain":
            st.session_state.cv_text = uploaded_cv_file.read().decode("utf-8")
        st.session_state.cv_file = uploaded_cv_file.name
    else:
        st.session_state.cv_text = cv_text_input

    if uploaded_job_file:
        if uploaded_job_file.type == "application/pdf":
            st.session_state.job_text = extract_text_from_pdf(uploaded_job_file)
        elif uploaded_job_file.type == "text/plain":
            st.session_state.job_text = uploaded_job_file.read().decode("utf-8")
        st.session_state.job_file = uploaded_job_file.name
    else:
        st.session_state.job_text = job_text_input

    # Daten speichern
    save_data({
        'name': name,
        'lastname': lastname,
        'job_title': job_title,
        'job_description': job_description,
        'cv_text': st.session_state.cv_text,
        'job_text': st.session_state.job_text,
        'cv_file': st.session_state.cv_file,
        'job_file': st.session_state.job_file,
        'cv_text_input': cv_text_input,
        'job_text_input': job_text_input
    })

if st.session_state.cv_text and st.session_state.job_text:
    lebenslauf_clean = clean_text(st.session_state.cv_text)
    stellenbeschreibung_clean = clean_text(st.session_state.job_text)

    # TF-IDF Vektorisierung
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([lebenslauf_clean, stellenbeschreibung_clean])

    # Cosinus-Ähnlichkeit berechnen
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    score = cosine_sim[0][0]

    # Ergebnis anzeigen
    st.write(f'Der Ähnlichkeits-Score zwischen Lebenslauf und Stellenbeschreibung beträgt: {score:.2f}')

    # TF-IDF Daten extrahieren
    labels = vectorizer.get_feature_names_out()
    cv_tfidf = tfidf_matrix.toarray()[0]
    job_tfidf = tfidf_matrix.toarray()[1]

    # Top N Schieberegler
    top_n = st.slider('Wählen Sie die Top N Begriffe aus', min_value=1, max_value=len(labels), value=10)

    # Sortiere nach den höchsten TF-IDF Werten und wähle die Top N
    top_n_indices = np.argsort(cv_tfidf + job_tfidf)[-top_n:]
    top_labels = labels[top_n_indices]
    top_cv_tfidf = cv_tfidf[top_n_indices]
    top_job_tfidf = job_tfidf[top_n_indices]

    # Balkendiagramm
    x = np.arange(len(top_labels))  # die Label-Positionen
    width = 0.35  # die Breite der Balken

    fig1, ax1 = plt.subplots(figsize=(10, 8))
    rects1 = ax1.bar(x - width / 2, top_cv_tfidf, width, label='Lebenslauf')
    rects2 = ax1.bar(x + width / 2, top_job_tfidf, width, label='Stellenbeschreibung')

    ax1.set_xlabel('Begriffe')
    ax1.set_ylabel('TF-IDF Wert')
    ax1.set_title('Top N TF-IDF Vektor Vergleich')
    ax1.set_xticks(x)
    ax1.set_xticklabels(top_labels, rotation=90)
    ax1.legend()

    fig1.tight_layout()

    st.pyplot(fig1)

    # Radarplot
    radar_fig = create_radar_chart(top_labels, top_cv_tfidf, top_job_tfidf)
    st.pyplot(radar_fig)

    # Ähnlichkeitsmatrix
    sim_matrix = pd.DataFrame(cosine_similarity(tfidf_matrix), index=['Lebenslauf', 'Stellenbeschreibung'],
                              columns=['Lebenslauf', 'Stellenbeschreibung'])
    st.write("Ähnlichkeitsmatrix")
    st.dataframe(sim_matrix)

    # Heatmap
    st.write("Heatmap der Ähnlichkeitsmatrix")
    fig2, ax2 = plt.subplots()
    sns.heatmap(sim_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax2)
    st.pyplot(fig2)

    # Dendrogramm
    st.write("Dendrogramm")
    linked = linkage(tfidf_matrix.toarray(), 'single')
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    dendrogram(linked, labels=['Lebenslauf', 'Stellenbeschreibung'], ax=ax3)
    st.pyplot(fig3)

    # Scatter Plot
    st.write("Scatter Plot der TF-IDF Werte")
    fig4, ax4 = plt.subplots()
    ax4.scatter(cv_tfidf, job_tfidf)
    ax4.set_xlabel('Lebenslauf TF-IDF')
    ax4.set_ylabel('Stellenbeschreibung TF-IDF')
    for i, label in enumerate(top_labels):
        ax4.annotate(label, (cv_tfidf[i], job_tfidf[i]))
    st.pyplot(fig4)

    # PCA Plot
    st.write("PCA Plot der TF-IDF Matrix")
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(tfidf_matrix.toarray())
    fig5, ax5 = plt.subplots()
    ax5.scatter(pca_result[:, 0], pca_result[:, 1])
    for i, label in enumerate(['Lebenslauf', 'Stellenbeschreibung']):
        ax5.annotate(label, (pca_result[i, 0], pca_result[i, 1]))
    st.pyplot(fig5)

    # TSNE Plot
    st.write("t-SNE Plot der TF-IDF Matrix")
    tsne = TSNE(n_components=2, perplexity=1, random_state=42)
    tsne_result = tsne.fit_transform(tfidf_matrix.toarray())
    fig6, ax6 = plt.subplots()
    ax6.scatter(tsne_result[:, 0], tsne_result[:, 1])
    for i, label in enumerate(['Lebenslauf', 'Stellenbeschreibung']):
        ax6.annotate(label, (tsne_result[i, 0], tsne_result[i, 1]))
    st.pyplot(fig6)

    # Wordclouds
    st.write("Wordcloud für den Lebenslauf")
    wordcloud_cv = create_wordcloud(' '.join([word for word, tfidf in zip(labels, cv_tfidf) if tfidf > 0]))
    fig7, ax7 = plt.subplots(figsize=(10, 5))
    ax7.imshow(wordcloud_cv, interpolation='bilinear')
    ax7.axis('off')
    st.pyplot(fig7)

    st.write("Wordcloud für die Stellenbeschreibung")
    wordcloud_job = create_wordcloud(' '.join([word for word, tfidf in zip(labels, job_tfidf) if tfidf > 0]))
    fig8, ax8 = plt.subplots(figsize=(10, 5))
    ax8.imshow(wordcloud_job, interpolation='bilinear')
    ax8.axis('off')
    st.pyplot(fig8)

    # PDF erstellen und Download-Link anzeigen
    figures = {
        "Top N TF-IDF Vektor Vergleich": save_fig_to_tempfile(fig1),
        "Radarplot": save_fig_to_tempfile(radar_fig),
        "Heatmap der Ähnlichkeitsmatrix": save_fig_to_tempfile(fig2),
        "Dendrogramm": save_fig_to_tempfile(fig3),
        "Scatter Plot der TF-IDF Werte": save_fig_to_tempfile(fig4),
        "PCA Plot der TF-IDF Matrix": save_fig_to_tempfile(fig5),
        "t-SNE Plot der TF-IDF Matrix": save_fig_to_tempfile(fig6),
        "Wordcloud für den Lebenslauf": save_fig_to_tempfile(fig7),
        "Wordcloud für die Stellenbeschreibung": save_fig_to_tempfile(fig8),
    }

    pdf_bytes = create_pdf(name, lastname, job_title, job_description, score, figures)
    st.download_button("Download PDF", data=pdf_bytes, file_name="report.pdf", mime="application/pdf")
else:
    st.error("Bitte geben Sie entweder den Text direkt ein oder laden Sie eine Datei hoch.")
