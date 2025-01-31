# Content Relevance Checker

This Streamlit application allows users to check the relevance of uploaded content (text, images, or videos) to a user-provided description. It leverages deep learning models to assess the semantic similarity between the content and the description.  *(Note: NLP preprocessing with spaCy has been removed from this version.)*

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [How It Works](#how-it-works)
    - [Text Content](#text-content)
    - [Image Content](#image-content)
    - [Video Content (Placeholder)](#video-content-placeholder)
- [Installation](#installation)
- [Usage](#usage)
- [Model Selection](#model-selection)
- [Limitations](#limitations)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Introduction

In today's information-saturated world, quickly determining the relevance of content is a critical task. This application simplifies this process by using advanced techniques to understand the meaning of both the uploaded content and the user's description. It then calculates the semantic similarity between them to provide a relevance assessment.

## Features

*   **Multi-Content Support:** Handles text, image, and (partially implemented) video uploads.
*   **Semantic Similarity:** Employs sentence transformer models to go beyond simple keyword matching and understand the *meaning* of the content and description.
*   **Clear Output:** Provides a relevance assessment (relevant/irrelevant) along with a confidence score indicating the model's certainty.
*   **Easy-to-Use Interface:** Built with Streamlit for a simple and intuitive user experience.

## How It Works

The application follows these steps:

1.  **User Upload:** The user uploads text, an image, or a video file and provides a descriptive text input.

2.  **Content Processing:**
    *   **Text:** The uploaded text is directly used for analysis.
    *   **Image:** The uploaded image is classified using an image classification model. The top predicted labels are then used for semantic comparison.
    *   **Video:** (Currently a placeholder) The video is displayed, but video analysis is not yet implemented.

3.  **Semantic Similarity Calculation:** Sentence transformer models are used to create embeddings (vector representations) of the description and the content (or image labels). The cosine similarity between these embeddings is calculated to determine how semantically related they are.

4.  **Relevance Assessment:** Based on the similarity score and a predefined threshold, the application determines if the content is relevant to the description. A confidence score (equal to the similarity) is also provided.

### Text Content

For text uploads, the text is directly used and then compared to the description.

### Image Content

For image uploads, the image is first classified using an image classification model. The top predicted labels from the image classifier are then used in the semantic similarity calculation against the description. This approach allows for assessing the relevance of an image based on its content.

### Video Content (Placeholder)

Currently, video uploads are only displayed. Full video analysis (frame extraction, image classification of frames, and combining results) is planned for future development.

## Installation

1.  **Clone the repository (if applicable):** If you're working with a Git repository, clone it to your local machine.

2.  **Create a virtual environment (recommended):**
    ```bash
    python3 -m venv .venv  # Or python -m venv .venv on Windows
    source .venv/bin/activate  # Activate the environment (Linux/macOS)
    .venv\Scripts\activate  # Activate the environment (Windows)
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```

2.  **Open the app in your browser:** Streamlit will provide you with a URL to access the app (usually `http://localhost:8501`).

3.  **Upload content and provide a description:** Use the file uploader to select your text, image, or video file. Enter a description in the text area.

4.  **Check relevance:** Click the "Check Relevance" button. The app will display the relevance assessment and confidence score.

## Model Selection

*   **Image Classification:** The application uses `google/vit-base-patch16-224` as the default image classification model. You can change this in the code to any other model available on the Hugging Face Model Hub. Vision Transformer (ViT) models are generally recommended for image classification tasks.

*   **Sentence Transformer:** The application uses `all-mpnet-base-v2` as the default sentence transformer model. You can experiment with other models from the `sentence-transformers` library, such as `all-MiniLM-L6-v2` (for faster processing) or `multi-qa-mpnet-base-dot-v1` (for question-answering-like scenarios).

## Limitations

*   **Video Analysis:** Video analysis is not yet fully implemented.
*   **Image Classification Accuracy:** The accuracy of the image classification depends on the chosen model and the quality of the uploaded image. Complex or unusual images might not be classified correctly.
*   **Semantic Similarity:** Semantic similarity is a complex problem. While the chosen models are effective, they might not always perfectly capture the nuances of human language.
*   **Computational Resources:** Running large language models can require significant computational resources (especially RAM and GPU).

## Future Enhancements

*   **Full Video Analysis:** Implement frame extraction and image classification for video uploads.
*   **Improved Image Classification:** Explore more specialized image classification models for specific domains.
*   **Fine-tuning:** Fine-tune the models on a custom dataset to improve accuracy for specific use cases.
*   **User Feedback:** Add a mechanism for users to provide feedback on the relevance assessments to further improve the model.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request if you have any suggestions or improvements.

## License

MIT License

## Contact

Shaik Dadapeer
shaikdadapeer4488@gmail.com