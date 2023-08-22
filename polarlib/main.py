import spacy

from polarlib.polar.news_corpus_collector import *
from polarlib.polar.actor_extractor import *
from polarlib.polar.topic_identifier import *
from polarlib.polar.sentiment_attitude_pipeline import *
from polarlib.polar.coalitions_and_conflicts import *
from polarlib.polar.sag_generator import *

if __name__ == "__main__":

    output_dir       = "/tmp/example"
    keywords         = ["openai", "gpt"]
    nlp              = spacy.load("en_core_web_sm")

    ##########################
    # News Corpus Collection #
    ##########################

    corpus_collector = NewsCorpusCollector(
        output_dir   = output_dir,
        from_date    = date(year = 2023, month = 8, day = 15),
        to_date      = date(year = 2023, month = 8, day = 15),
        keywords     = keywords
    )

    corpus_collector.collect_archives()
    corpus_collector.collect_articles()
    corpus_collector.pre_process_articles()

    ############################
    # Entity and NP Extraction #
    ############################

    entity_extractor = EntityExtractor(output_dir   = output_dir)

    entity_extractor.extract_entities()

    noun_phrase_extractor = NounPhraseExtractor(output_dir=output_dir)

    noun_phrase_extractor.extract_noun_phrases()

    ###################################
    # Discussion Topic Identification #
    ###################################

    topic_identifier = TopicIdentifier(output_dir=output_dir)
    topic_identifier.encode_noun_phrases()
    topic_identifier.noun_phrase_clustering()

    #####################################
    # Sentiment Attitude Classification #
    #####################################

    batch_size                  = 8
    gradient_accumulation_steps = 4
    logging_steps               = 10

    training_args = TrainingArguments(
        output_dir                  = '../.cache',
        weight_decay                = 0.02,
        learning_rate               = 1e-5,
        evaluation_strategy         = "epoch",
        save_strategy               = 'epoch',
        num_train_epochs            = 10,
        logging_steps               = logging_steps,
        load_best_model_at_end      = True,
        save_total_limit            = 2,
        per_device_train_batch_size = batch_size,
        per_device_eval_batch_size  = batch_size,
        gradient_accumulation_steps = gradient_accumulation_steps,
        disable_tqdm                = True
    )

    model_path = "/home/dpasch01/notebooks/Sentiment Attitude Classification/models/roberta-base-sentiment-attitude/pretrained"

    model_name = 'roberta-base'

    tokenizer  = AutoTokenizer.from_pretrained(model_name)
    model      = AutoModelForSequenceClassification.from_pretrained(model_path)

    sentiment_attitude_pipeline = SentimentAttitudePipeline(
        output_dir    = output_dir,
        model         = model,
        tokenizer     = tokenizer,
        training_args = training_args
    )

    sentiment_attitude_pipeline.calculate_sentiment_attitudes()

    #########################################
    # Sentiment Attitude Graph Construction #
    #########################################

    sag_generator = SAGGenerator(output_dir)

    sag_generator.load_sentiment_attitudes()

    bins = sag_generator.calculate_attitude_buckets(verbose=True)

    sag_generator.convert_attitude_signs(
        bin_category_mapping = {
            "NEGATIVE": [bins[0], bins[1], bins[2], bins[3]],
            "NEUTRAL":  [bins[4], bins[5]],
            "POSITIVE": [bins[6], bins[7], bins[8], bins[9]]
        },
        verbose              = True
    )

    sag_generator.construct_sag()

    ################################
    # Entity Fellowship Extraction #
    ################################

    fellowship_extractor = FellowshipExtractor(output_dir)

    fellowships          = fellowship_extractor.extract_fellowships(
        n_iter     = 1,
        resolution = 0.05,
        merge_iter = 1,
        jar_path   ='./',
        verbose    = True
    )

    ################################
    # Fellowship Dipole Generation #
    ################################

    dipole_generator = DipoleGenerator(output_dir)
    dipoles          = dipole_generator.generate_dipoles(f_g_thr=0.7, n_r_thr=0.5)

    #############################
    # Dipole Topic Polarization #
    #############################

    topic_attitude_calculator = TopicAttitudeCalculator(output_dir)
    topic_attitude_calculator.load_sentiment_attitudes()

    topic_attitudes  = topic_attitude_calculator.get_topic_attitudes()