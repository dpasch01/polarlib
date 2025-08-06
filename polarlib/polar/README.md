## Polarlib / POLAR

An end-to-end pipeline for unsupervised extraction and modeling of domain-specific Polarization Knowledge Graphs (PKGs) from news articles.

## Usage

### POLAR Quickstart

![POLAR Framework Architecture](polar.png)

*Figure 2: POLAR architecture for the unsupervised and domain-agnostic extraction of polarization knowledge from news articles.*

The main components of POLAR include:

- News Corpus Collector: Collects news articles related to specific keywords and time frames.
- Entity and Noun Phrase Extraction: Extracts entities and noun phrases from collected articles.
- Discussion Topic Identification: Identifies discussion topics based on extracted noun phrases.
- Sentiment Attitude Classification: Classifies sentiment attitudes using syntactical methods.
- Sentiment Attitude Graph Generation: Generates sentiment attitude graphs for analysis.
- Entity Fellowship Extraction: Extracts entity fellowships from generated graphs.
- Fellowship Dipole Generation: Generates fellowship dipoles based on extracted fellowships.
- Dipole Topic Polarization: Calculates topic polarization based on generated dipoles.

To use the `polarlib` library, follow these steps:

1. Clone the repository: `git clone https://github.com/dpasch01/polarlib.git`
2. Install the dependencies: `pip install -t requirements.txt`
3. Use the code snippets below:

```python
from polarlib.polar.attitude.syntactical_sentiment_attitude import SyntacticalSentimentAttitudePipeline
from polarlib.polar.news_corpus_collector import *
from polarlib.polar.actor_extractor import *
from polarlib.polar.topic_identifier import *
from polarlib.polar.coalitions_and_conflicts import *
from polarlib.polar.sag_generator import *

# Example Workflow
OUTPUT_DIR = "/tmp/example"
keywords = ["openai", "altman", 'chatgpt', 'gpt']

# Collect news corpus
corpus_collector = NewsCorpusCollector(
    output_dir = OUTPUT_DIR,
    from_date = date(year=2023, month=11, day=16),
    to_date = date(year=2023, month=11, day=23),
    keywords = keywords
)

corpus_collector.collect_archives()
corpus_collector.collect_articles(n_articles = 250)
corpus_collector.pre_process_articles()

# Extract entities and noun phrases
entity_extractor = EntityExtractor(output_dir=OUTPUT_DIR, coref=False)
entity_extractor.extract_entities()

noun_phrase_extractor = NounPhraseExtractor(output_dir=OUTPUT_DIR)
noun_phrase_extractor.extract_noun_phrases()

# Extract the discussion topics
topic_identifier = TopicIdentifier(output_dir=OUTPUT_DIR, llama_wv=False)
topic_identifier.encode_noun_phrases()
topic_identifier.noun_phrase_clustering(threshold=0.8)

# Calculate the attitudes between entities and topics
sentiment_attitude_pipeline = SyntacticalSentimentAttitudePipeline(
	output_dir  = OUTPUT_DIR,
        nlp = spacy.load("en_core_web_sm"),
	dictionary_path = "mpqa.csv"
)

sentiment_attitude_pipeline.calculate_sentiment_attitudes()

# Construct the Sentiment Attitude Graph (SAG)
sag_generator = SAGGenerator(OUTPUT_DIR)

sag_generator.load_sentiment_attitudes()

bins = sag_generator.calculate_attitude_buckets(verbose=True, figsize=(16, 4))

sag_generator.convert_attitude_signs(
    bin_category_mapping = {
        "NEGATIVE": [(-1.00, -0.02)],
        "NEUTRAL": [(-0.02, 0.02)],
        "POSITIVE": [(0.02, 1.00)]
    },
    minimum_frequency = 5,
    verbose = True
)

G, node_to_int, int_to_node = sag_generator.construct_sag()

# Extract the fellowships and dipoles
fellowship_extractor = FellowshipExtractor(OUTPUT_DIR)

fellowships = fellowship_extractor.extract_fellowships(
    n_iter = 10,
    resolution = 0.075,
    merge_iter = 10,
    jar_path ='./polarlib',
    verbose = True,
    output_flag = True
)

dipole_generator = DipoleGenerator(OUTPUT_DIR)
dipoles = dipole_generator.generate_dipoles(f_g_thr=0.7, n_r_thr=0.5)

# Calculate topic polarization
topic_attitude_calculator = TopicAttitudeCalculator(OUTPUT_DIR)

topic_attitude_calculator.load_sentiment_attitudes()

dipole_topics_dict = topic_attitude_calculator.get_polarization_topics()

topic_attitudes = topic_attitude_calculator.get_topic_attitudes()

```
