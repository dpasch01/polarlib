# Polarlib Library

The `polarlib` library provides a comprehensive set of tools and functionalities for multi-level polarization analysis and sentiment attitude classification. This library offers modules and classes that enable users to process, analyze, and visualize polarization-related data. Here's a brief overview of the library's main components:

## Table of Contents

- [Introduction](#introduction)
- [Usage](#usage)
- [Components](#components)
- [Examples](#examples)

## Introduction

The `polarlib` library is designed to facilitate the modeling, analysis, and visualization of multi-level polarization and sentiment attitude data. It provides a collection of modules and classes that cover various aspects of the polarization analysis pipeline, including data collection, entity extraction, topic identification, sentiment attitude classification, graph generation, and more.

## Usage

To utilize the `polarlib` library, you can import the necessary modules and classes. Below is an example of how to use different components of the library:

```python
from polarlib.polar.attitude.syntactical_sentiment_attitude import SyntacticalSentimentAttitudePipeline
from polarlib.polar.news_corpus_collector import *
from polarlib.polar.actor_extractor import *
from polarlib.polar.topic_identifier import *
from polarlib.polar.coalitions_and_conflicts import *
from polarlib.polar.sag_generator import *

if __name__ == "__main__":
    # Initialize variables
    output_dir = "/tmp/example"
    keywords = ["openai", "gpt"]
    nlp = spacy.load("en_core_web_sm")

    # News Corpus Collection
    corpus_collector = NewsCorpusCollector(output_dir=output_dir, ...)

    # Entity and NP Extraction
    entity_extractor = EntityExtractor(output_dir=output_dir)
    noun_phrase_extractor = NounPhraseExtractor(output_dir=output_dir)

    # Discussion Topic Identification
    topic_identifier = TopicIdentifier(output_dir=output_dir)

    # Sentiment Attitude Classification
    sentiment_attitude_pipeline = SyntacticalSentimentAttitudePipeline(output_dir=output_dir, nlp=nlp, ...)

    # Sentiment Attitude Graph Construction
    sag_generator = SAGGenerator(output_dir)
    sag_generator.load_sentiment_attitudes()
    sag_generator.convert_attitude_signs(bin_category_mapping={}, ...)

    # Entity Fellowship Extraction
    fellowship_extractor = FellowshipExtractor(output_dir)
    fellowships = fellowship_extractor.extract_fellowships(...)

    # Fellowship Dipole Generation
    dipole_generator = DipoleGenerator(output_dir)
    dipoles = dipole_generator.generate_dipoles(...)

    # Dipole Topic Polarization
    topic_attitude_calculator = TopicAttitudeCalculator(output_dir)
    topic_attitude_calculator.load_sentiment_attitudes()
    topic_attitudes = topic_attitude_calculator.get_topic_attitudes()
```

## Components

The `polarlib` library consists of various components that address different aspects of multi-level polarization analysis. These components include:

- News Corpus Collector: Collects news articles related to specific keywords and time frames.
- Entity and Noun Phrase Extraction: Extracts entities and noun phrases from collected articles.
- Discussion Topic Identification: Identifies discussion topics based on extracted noun phrases.
- Sentiment Attitude Classification: Classifies sentiment attitudes using syntactical methods.
- Sentiment Attitude Graph Generation: Generates sentiment attitude graphs for analysis.
- Entity Fellowship Extraction: Extracts entity fellowships from generated graphs.
- Fellowship Dipole Generation: Generates fellowship dipoles based on extracted fellowships.
- Dipole Topic Polarization: Calculates topic polarization based on generated dipoles.

## Examples

For more detailed examples and usage instructions, please refer to the code snippets and documentation for each component within the library.

If you have any questions or need further assistance, please don't hesitate to reach out to us.

**Note:** The library requires external dependencies such as the `spacy` natural language processing library and relevant data sources. Make sure to install these dependencies and provide the required inputs for accurate analysis.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
