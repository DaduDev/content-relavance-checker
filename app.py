import streamlit as st
from transformers import pipeline
from PIL import Image  # For image handling
import io  # For video handling (more on this later)
from sentence_transformers import SentenceTransformer, util
import spacy  # For NLP processing
import torch  # For tensor operations


# ... (Classifier initialization and classify_post function remain the same)
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")  # Or a similar model
sentence_model = SentenceTransformer('all-mpnet-base-v2')
nlp = spacy.load("en_core_web_lg")
def process_description(description):
    doc = nlp(description)

    # 1. Lemmatization: Get the base form of words
    lemmatized_description = " ".join([token.lemma_ for token in doc])

    # 2. Stop Word Removal (Optional, but often helpful):
    stop_words = spacy.lang.en.stop_words.STOP_WORDS
    filtered_description = " ".join([token.text for token in doc if not token.is_stop])

    # 3. Combining (You can experiment with different combinations)
    processed_description = lemmatized_description # Or filtered_description, or both

    return processed_description

def classify_post(post_text, description):
    keywords = description.lower().split()  # Extract keywords
    text = post_text.lower()

    if not any(keyword in text for keyword in keywords):  # Check for keyword matches
        return "irrelevant", 0.99  # High confidence for irrelevance if no keywords match

    # If keywords ARE present, then use the zero-shot model:
    try:
        result = classifier(post_text, candidate_labels=["relevant", "irrelevant"], hypothesis_template=f"This text is specifically about or directly related to the topic of {description}.")
        predicted_label = result['labels'][0]
        confidence = result['scores'][0]
        return predicted_label, confidence
    except Exception as e:
        return f"Classification error: {e}", 0.0

def main():
    st.title("Content Relevance Checker")

    uploaded_file = st.file_uploader("Upload text, image, or video", type=["txt", "jpg", "png", "mp4", "mov"])  # Allow multiple file types
    description = st.text_area("Enter a description/topic:")
    processed_description = process_description(description)  # Process the description!
    description_embedding = sentence_model.encode(processed_description)  # Use processed description


    if st.button("Check Relevance"):
        if not uploaded_file or not description:
            st.warning("Please upload a file and enter a description.")
            return

        file_type = uploaded_file.type

        if file_type.startswith("text"):
            text_content = uploaded_file.getvalue().decode("utf-8")  # Decode text
            st.subheader("Uploaded Text:")
            st.write(text_content)

            predicted_label, confidence = classify_post(text_content, description) # Classify text

        elif file_type.startswith("image"):
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # For image classification, you'll need a different model.  Example:
            image_classifier = pipeline("image-classification", model="google/vit-base-patch16-224") # Or a more specific model
            try:
                image_result = image_classifier(image)
                st.write("Image Classification Result:")
                st.write(image_result)

                # Semantic Similarity Check:
                description_embedding = sentence_model.encode(description)

                best_similarity = 0
                best_label = None

                for result in image_result:
                    label = result['label']
                    label_embedding = sentence_model.encode(label)
                    similarity = util.cos_sim(description_embedding, label_embedding)  # Cosine similarity
                    #print(f"{label}: {similarity}") # For Debugging
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_label = label

                st.write(f"Best Matching Label: {best_label} (Similarity: {best_similarity})")

                if best_similarity > 0.5:
                    predicted_label = "relevant"
                    confidence = best_similarity.item()  # Convert Tensor to float
                else:
                    predicted_label = "irrelevant"
                    confidence = (1 - best_similarity).item() # Convert Tensor to float

                st.write(f"Confidence: {confidence:.2f}")  # Now this will work

            except Exception as e:
                st.error(f"Image classification error: {e}")
                return

        elif file_type.startswith("video"): # Very basic video handling
            video_bytes = uploaded_file.getvalue()
            st.video(video_bytes)

            # Video analysis is much more complex.  You would typically need to extract frames, process them individually (image classification), and then combine the results.  This is beyond the scope of a simple example.
            st.write("Video analysis is not yet implemented in this example.")
            return

        else:
            st.warning("Unsupported file type.")
            return

        st.subheader("Relevance Assessment:")
        st.write(f"The uploaded content is **{predicted_label}** to the topic of '{description}'.")
        st.write(f"Confidence: {confidence:.2f}")

        if predicted_label == "irrelevant":
            st.write("Consider refining your search or description.")
        elif confidence < 0.7:
            st.write("The model is not very confident in its prediction. The content might be marginally relevant or ambiguous.")

if __name__ == "__main__":
    main()