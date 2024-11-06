from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import yaml
import nltk
import string

ps = PorterStemmer()
def dictionary_features(text):

    dictionary_feature = {}
    number_of_affiliation_words = 0
    number_of_bad_words = 0
    number_of_certainty_words = 0
    number_of_common_words = 0
    number_of_comparison_words = 0
    number_of_concern_words = 0
    number_of_emotional_tone_words = 0
    number_of_emotion_words = 0
    number_of_interrogative_words = 0
    number_of_negative_words = 0
    number_of_power_words = 0
    number_of_quantity_words = 0
    number_of_reward_words = 0
    number_of_risk_words = 0
    number_of_slang_words = 0
    number_of_analytical_words = 0
    number_of_negative_words_Hu_Liu = 0
    number_of_positive_words_Hu_Liu = 0
    AFINN_word_score = 0


    text = text.lower()

    #Affiliationwords
    with open('dictionaries/affiliationwords.txt' , 'r') as textfile:
        for line in textfile:
            ps.stem(line)
            if text.find(line.lower()) != -1:
                number_of_affiliation_words = number_of_affiliation_words + 1

    #analyticalwords
    with open('dictionaries/analyticalwords.txt' , 'r') as textfile:
        for line in textfile:
            ps.stem(line)
            if text.find(line.lower()) != -1:
                number_of_analytical_words = number_of_analytical_words + 1

    #Badwords
    with open('dictionaries/badwords.txt' , 'r') as textfile:
        for line in textfile:
            ps.stem(line)
            if text.find(line.lower()) != -1:
                number_of_bad_words = number_of_bad_words + 1

    #Certaintywords
    with open('dictionaries/certainty.txt' , 'r') as textfile:
        for line in textfile:
            ps.stem(line)
            if text.find(line.lower()) != -1:
                number_of_certainty_words = number_of_certainty_words + 1

    #Comparisonwords
    with open('dictionaries/comparisonwords.txt' , 'r') as textfile:
        for line in textfile:
            ps.stem(line)
            if text.find(line.lower()) != -1:
                number_of_comparison_words = number_of_comparison_words + 1

    #commonwords
    with open('dictionaries/commonwords.txt' , 'r') as textfile:
        for line in textfile:
            ps.stem(line)
            if text.find(line.lower()) != -1:
                number_of_common_words = number_of_common_words + 1

    #concernwords
    with open('dictionaries/concernwords.txt' , 'r') as textfile:
        for line in textfile:
            ps.stem(line)
            if text.find(line.lower()) != -1:
                number_of_concern_words = number_of_concern_words + 1

    #emotionaltonewords
    with open('dictionaries/emotionaltonewords.txt' , 'r') as textfile:
        for line in textfile:
            ps.stem(line)
            if text.find(line.lower()) != -1:
                number_of_emotional_tone_words = number_of_emotional_tone_words + 1

    #emotionwords
    with open('dictionaries/emotionwords.txt' , 'r') as textfile:
        for line in textfile:
            ps.stem(line)
            if text.find(line.lower()) != -1:
                number_of_emotion_words = number_of_emotion_words + 1

    #interrogativewords
    with open('dictionaries/interrogativewords.txt' , 'r') as textfile:
        for line in textfile:
            ps.stem(line)
            if text.find(line.lower()) != -1:
                number_of_interrogative_words = number_of_interrogative_words + 1

    #negativewords
    with open('dictionaries/negativewords.txt' , 'r') as textfile:
        for line in textfile:
            ps.stem(line)
            if text.find(line.lower()) != -1:
                number_of_negative_words = number_of_negative_words + 1

   #powerwords
    with open('dictionaries/powerwords.txt' , 'r') as textfile:
        for line in textfile:
            ps.stem(line)
            if text.find(line.lower()) != -1:
                number_of_power_words = number_of_power_words + 1

    #quantitywords
    with open('dictionaries/quantitywords.txt', 'r') as textfile:
        for line in textfile:
            ps.stem(line)
            if text.find(line.lower()) != -1:
                number_of_quantity_words = number_of_quantity_words + 1

    #rewardwords
    with open('dictionaries/rewardwords.txt', 'r') as textfile:
        for line in textfile:
            ps.stem(line)
            if text.find(line.lower()) != -1:
                number_of_reward_words = number_of_reward_words + 1

    #riskwords
    with open('dictionaries/riskwords.txt', 'r') as textfile:
        for line in textfile:
            ps.stem(line)
            if text.find(line.lower()) != -1:
                number_of_risk_words = number_of_risk_words + 1

    #slangwords
    with open('dictionaries/slangwords.txt', 'r') as textfile:
        for line in textfile:
            ps.stem(line)
            if text.find(line.lower()) != -1:
                number_of_slang_words = number_of_slang_words + 1

    #negative_words_Hu_Liu
    with open('dictionaries/Hu-LiuNegative.txt', 'r') as textfile:
        for line in textfile:
            ps.stem(line)
            if text.find(line.lower()) != -1:
                number_of_negative_words_Hu_Liu = number_of_negative_words_Hu_Liu + 1

    #positive_words_Hu_Liu
    with open('dictionaries/Hu-LiuPositive.txt', 'r') as textfile:
        for line in textfile:
            line=ps.stem(line)
            if text.find(line.lower()) != -1:
                number_of_positive_words_Hu_Liu = number_of_positive_words_Hu_Liu + 1

    #AFINN_word_score
    with open('dictionaries/AFINN.txt', 'r') as textfile:
        for line in textfile:
            afinn = line.split('\t')
            afinn[0] = ps.stem(afinn[0])
            if text.find(afinn[0].lower()) != -1:
                AFINN_word_score = AFINN_word_score + int(afinn[1])
        if len(text.split()) != 0:
            AFINN_word_score = AFINN_word_score/ len(text.split())
        else:
            AFINN_word_score = 0

    with open("dictionaries/Loughran-McDonald.yml", 'r') as stream:
        try:
            yamlfile= yaml.load(stream)
            for i in yamlfile:
                for key in i:
                    dictionary_feature[key] = 0
                    count = 0
                    for words in i.get(key):
                        words = ps.stem(words)
                        if text.find(words.lower()) != -1:
                            #print (key+"\t"+words)
                            count = count + 1
                    dictionary_feature["LM_"+key] = count / len(text.split())

        except yaml.YAMLError as exc:
            print(exc)

    with open("dictionaries/laver-garry.yml", 'r') as stream:
        try:
            yamlfile= yaml.load(stream)
            for i in yamlfile:
                for key in i:
                    dictionary_feature[key] = 0
                    count = 0
                    for words in i.get(key):
                        if isinstance(words,str):
                            words = ps.stem(words)
                            if text.find(words.lower()) != -1:
                                count = count + 1
                        else:
                            for key1 in words:
                                dictionary_feature[key1] = 0
                                count1 = 0
                                for words1 in words.get(key1):
                                    words1 = ps.stem(words1)
                                    if text.find(words1.lower()) != -1:
                                        count1 = count1 + 1

                                dictionary_feature["LG_"+key1] = count1/len(text.split())
                                count = count + count1

                    dictionary_feature["LG_"+key] = count/ len(text.split())

        except yaml.YAMLError as exc:
            print(exc)

    with open("dictionaries/RID.yml", 'r') as stream:
        try:
            yamlfile= yaml.load(stream)
            for i in yamlfile:
                for key in i:
                    dictionary_feature[key] = 0
                    count = 0
                    for words in i.get(key):
                        if isinstance(words,str):
                            words = ps.stem(words)
                            if text.find(words.lower()) != -1:
                                count = count + 1
                        else:
                            for key1 in words:
                                dictionary_feature[key1] = 0
                                count1 = 0
                                for words1 in words.get(key1):
                                    if isinstance(words1, str):
                                        words1 = ps.stem(words1)
                                        if text.find(words1.lower()) != -1:
                                            count1 = count1 + 1
                                    else:
                                        for key2 in words1:
                                            dictionary_feature[key2] = 0
                                            count2 = 0
                                            for words2 in words1.get(key2):
                                                words2 = ps.stem(words2)
                                                if text.find(words2.lower()) != -1:
                                                    count2 = count2 + 1

                                            dictionary_feature["RID_"+key2] = count2 / len(text.split())
                                            count1 = count1 + count2

                                dictionary_feature["RID_"+key1] = count1/len(text.split())
                                count = count + count1

                    dictionary_feature["RID_"+key] = count/ len(text.split())

        except yaml.YAMLError as exc:
            print(exc)

    #Adding to dictionary
    dictionary_feature['number_of_future_tense_words'] = number_of_affiliation_words
    dictionary_feature['number_of_analytical_words'] = number_of_analytical_words
    dictionary_feature['number_of_bad_words'] = number_of_bad_words
    dictionary_feature['number_of_certainty_words'] = number_of_certainty_words
    dictionary_feature['number_of_comparison_words'] = number_of_comparison_words
    dictionary_feature['number_of_common_words'] = number_of_common_words
    dictionary_feature['number_of_concern_words'] = number_of_concern_words
    dictionary_feature['number_of_emotional_tone_words'] = number_of_emotional_tone_words
    dictionary_feature['number_of_emotion_words'] = number_of_emotion_words
    dictionary_feature['number_of_interrogative_words'] = number_of_interrogative_words
    dictionary_feature['number_of_negative_words'] = number_of_negative_words
    dictionary_feature['number_of_power_words'] = number_of_power_words
    dictionary_feature['number_of_quantity_words'] = number_of_quantity_words
    dictionary_feature['number_of_reward_words'] = number_of_reward_words
    dictionary_feature['number_of_risk_words'] = number_of_risk_words
    dictionary_feature['number_of_slang_words'] = number_of_slang_words
    dictionary_feature['number_of_negative_words_Hu_Liu'] = number_of_negative_words_Hu_Liu
    dictionary_feature['number_of_positive_words_Hu_Liu'] = number_of_positive_words_Hu_Liu
    dictionary_feature['AFINN_word_score'] = AFINN_word_score

    return dictionary_feature


def dictionary_features_v2(text):

    dictionary_feature = {}

    words = nltk.word_tokenize(text)
    original_words = [w.rstrip(string.punctuation).lstrip(string.punctuation).lower()  for w in words]
    original_words = [ps.stem(w) for w in original_words]
    #print(original_words)

    # variables definition
    dictionary_feature = {}
    number_of_affiliation_words = 0
    number_of_bad_words = 0
    number_of_certainty_words = 0
    number_of_common_words = 0
    number_of_comparison_words = 0
    number_of_concern_words = 0
    number_of_emotional_tone_words = 0
    number_of_emotion_words = 0
    number_of_interrogative_words = 0
    number_of_negative_words = 0
    number_of_power_words = 0
    number_of_quantity_words = 0
    number_of_reward_words = 0
    number_of_risk_words = 0
    number_of_slang_words = 0
    number_of_analytical_words = 0
    number_of_negative_words_Hu_Liu = 0
    number_of_positive_words_Hu_Liu = 0
    AFINN_word_score = 0

    # Affiliationwords
    with open('dictionaries/affiliationwords.txt', 'r') as textfile:
        for line in textfile:
            line=ps.stem(line)
            if line in original_words:
                number_of_affiliation_words = number_of_affiliation_words + 1

    # analyticalwords
    with open('dictionaries/analyticalwords.txt', 'r') as textfile:
        for line in textfile:
            line=ps.stem(line)
            if len(line.split()) > 1:
                if text.find(line.lower()) != -1:
                    number_of_analytical_words = number_of_analytical_words + 1
            elif line in original_words:
                number_of_analytical_words = number_of_analytical_words + 1

    # Badwords
    with open('dictionaries/badwords.txt', 'r') as textfile:
        for line in textfile:
            line = ps.stem(line)
            if len(line.split()) > 1:
                if text.find(line.lower()) != -1:
                    number_of_bad_words = number_of_bad_words + 1
            elif line in original_words:
                number_of_bad_words = number_of_bad_words + 1

    # Certaintywords
    with open('dictionaries/certainty.txt', 'r') as textfile:
        for line in textfile:
            line = ps.stem(line)
            if len(line.split()) > 1:
                if text.find(line.lower()) != -1:
                    number_of_certainty_words = number_of_certainty_words + 1
            elif line in original_words:
                number_of_certainty_words = number_of_certainty_words + 1

    # Comparisonwords
    with open('dictionaries/comparisonwords.txt', 'r') as textfile:
        for line in textfile:
            line = ps.stem(line)
            if len(line.split()) > 1:
                if text.find(line.lower()) != -1:
                    number_of_comparison_words = number_of_comparison_words + 1
            elif line in original_words:
                number_of_comparison_words = number_of_comparison_words + 1

    # commonwords
    with open('dictionaries/commonwords.txt', 'r') as textfile:
        for line in textfile:
            line = ps.stem(line)
            if len(line.split()) > 1:
                if text.find(line.lower()) != -1:
                    number_of_common_words = number_of_common_words + 1
            elif line in original_words:
                number_of_common_words = number_of_common_words + 1

    # concernwords
    with open('dictionaries/concernwords.txt', 'r') as textfile:
        for line in textfile:
            line = ps.stem(line)
            if len(line.split()) > 1:
                if text.find(line.lower()) != -1:
                    number_of_concern_words = number_of_concern_words + 1
            elif line in original_words:
                number_of_concern_words = number_of_concern_words + 1

    # emotionaltonewords
    with open('dictionaries/emotionaltonewords.txt', 'r') as textfile:
        for line in textfile:
            line = ps.stem(line)
            if len(line.split()) > 1:
                if text.find(line.lower()) != -1:
                    number_of_emotional_tone_words = number_of_emotional_tone_words + 1
            elif line in original_words:
                number_of_emotional_tone_words = number_of_emotional_tone_words + 1

    # emotionwords
    with open('dictionaries/emotionwords.txt', 'r') as textfile:
        for line in textfile:
            line = ps.stem(line)
            if len(line.split()) > 1:
                if text.find(line.lower()) != -1:
                    number_of_emotion_words = number_of_emotion_words + 1
            elif line in original_words:
                number_of_emotion_words = number_of_emotion_words + 1

    # interrogativewords
    with open('dictionaries/interrogativewords.txt', 'r') as textfile:
        for line in textfile:
            line = ps.stem(line)
            if len(line.split()) > 1:
                if text.find(line.lower()) != -1:
                    number_of_interrogative_words = number_of_interrogative_words + 1
            elif line in original_words:
                number_of_interrogative_words = number_of_interrogative_words + 1

    # negativewords
    with open('dictionaries/negativewords.txt', 'r') as textfile:
        for line in textfile:
            line = ps.stem(line)
            if len(line.split()) > 1:
                if text.find(line.lower()) != -1:
                    number_of_negative_words = number_of_negative_words + 1
            elif line in original_words:
                number_of_negative_words = number_of_negative_words + 1

    # powerwords
    with open('dictionaries/powerwords.txt', 'r') as textfile:
        for line in textfile:
            line = ps.stem(line)
            if len(line.split()) > 1:
                if text.find(line.lower()) != -1:
                    number_of_power_words = number_of_power_words + 1
            elif line in original_words:
                number_of_power_words = number_of_power_words + 1

    # quantitywords
    with open('dictionaries/quantitywords.txt', 'r') as textfile:
        for line in textfile:
            line = ps.stem(line)
            if len(line.split()) > 1:
                if text.find(line.lower()) != -1:
                    number_of_quantity_words = number_of_quantity_words + 1
            elif line in original_words:
                number_of_quantity_words = number_of_quantity_words + 1

    # rewardwords
    with open('dictionaries/rewardwords.txt', 'r') as textfile:
        for line in textfile:
            line = ps.stem(line)
            if len(line.split()) > 1:
                if text.find(line.lower()) != -1:
                    number_of_reward_words = number_of_reward_words + 1
            elif line in original_words:
                number_of_reward_words = number_of_reward_words + 1

    # riskwords
    with open('dictionaries/riskwords.txt', 'r') as textfile:
        for line in textfile:
            line = ps.stem(line)
            if len(line.split()) > 1:
                if text.find(line.lower()) != -1:
                    number_of_risk_words = number_of_risk_words + 1
            elif line in original_words:
                number_of_risk_words = number_of_risk_words + 1

    # slangwords
    with open('dictionaries/slangwords.txt', 'r') as textfile:
        for line in textfile:
            line = ps.stem(line)
            if len(line.split()) > 1:
                if text.find(line.lower()) != -1:
                    number_of_slang_words = number_of_slang_words + 1
            elif line in original_words:
                number_of_slang_words = number_of_slang_words + 1

    # negative_words_Hu_Liu
    with open('dictionaries/Hu-LiuNegative.txt', 'r') as textfile:
        for line in textfile:
            line=ps.stem(line)
            if len(line.split()) > 1:
                if text.find(line.lower()) != -1:
                    number_of_negative_words_Hu_Liu = number_of_negative_words_Hu_Liu + 1
            elif line in original_words:
                number_of_negative_words_Hu_Liu = number_of_negative_words_Hu_Liu + 1

    # positive_words_Hu_Liu
    with open('dictionaries/Hu-LiuPositive.txt', 'r') as textfile:
        for line in textfile:
            line = ps.stem(line)
            if len(line.split()) > 1:
                if text.find(line.lower()) != -1:
                    number_of_positive_words_Hu_Liu = number_of_positive_words_Hu_Liu + 1
            elif line in original_words:
                number_of_positive_words_Hu_Liu = number_of_positive_words_Hu_Liu + 1

    # AFINN_word_score
    with open('dictionaries/AFINN.txt', 'r') as textfile:
        for line in textfile:
            afinn = line.split('\t')
            afinn[0] = ps.stem(afinn[0])
            if len(afinn[0].split()) > 1:
                if text.find(afinn[0].lower()) != -1:
                    AFINN_word_score = AFINN_word_score + int(afinn[1])
            elif afinn[0] in original_words:
                AFINN_word_score = AFINN_word_score + int(afinn[1])
        if len(text.split()) != 0:
            AFINN_word_score = AFINN_word_score / len(text.split())
        else:
            AFINN_word_score = 0

    with open("dictionaries/Loughran-McDonald.yml", 'r') as stream:
        try:
            yamlfile= yaml.load(stream)
            for i in yamlfile:
                for key in i:
                    count = 0
                    for words in i.get(key):
                        words = ps.stem(words)
                        words = words.replace("*", "")
                        if words in original_words:
                            count = count + 1
                    dictionary_feature["LM_"+key] = count / len(text.split())

        except yaml.YAMLError as exc:
            print(exc)

    with open("dictionaries/laver-garry.yml", 'r') as stream:
        try:
            yamlfile = yaml.load(stream)
            for i in yamlfile:
                for key in i:
                    count = 0
                    for words in i.get(key):
                        if isinstance(words, str):
                            words = ps.stem(words)
                            words = words.replace("*", "")
                            if words in original_words:
                                count = count + 1

                        else:
                            for key1 in words:
                                dictionary_feature[key1] = 0
                                count1 = 0
                                for words1 in words.get(key1):
                                    words1 = ps.stem(words1)
                                    words1 = words1.replace("*","")
                                    if words1 in original_words:
                                        count1 = count1 + 1

                                dictionary_feature["LG_" + key1] = count1 / len(text.split())
                                count = count + count1

                    dictionary_feature["LG_" + key] = count / len(text.split())

        except yaml.YAMLError as exc:
            print(exc)

    with open("dictionaries/RID.yml", 'r') as stream:
        try:
            yamlfile= yaml.load(stream)
            for i in yamlfile:
                for key in i:
                    count = 0
                    for words in i.get(key):
                        if isinstance(words,str):
                            words = ps.stem(words)
                            words = words.replace("*", "")
                            if words in original_words:
                                count = count + 1
                        else:
                            for key1 in words:
                                dictionary_feature[key1] = 0
                                count1 = 0
                                for words1 in words.get(key1):
                                    if isinstance(words1, str):
                                        words1 = ps.stem(words1)
                                        words1 = words1.replace("*", "")
                                        if words1 in original_words:
                                            count1 = count1 + 1

                                    else:
                                        for key2 in words1:
                                            dictionary_feature[key2] = 0
                                            count2 = 0
                                            for words2 in words1.get(key2):
                                                words2 = ps.stem(words2)
                                                words2 = words2.replace("*", "")
                                                if words2 in original_words:
                                                    count2 = count2 + 1

                                            dictionary_feature["RID_"+key2] = count2 / len(text.split())
                                            count1 = count1 + count2

                                dictionary_feature["RID_"+key1] = count1/len(text.split())
                                count = count + count1

                    dictionary_feature["RID_"+key] = count/ len(text.split())

        except yaml.YAMLError as exc:
            print(exc)


    #Adding to dictionary
    dictionary_feature['number_of_future_tense_words'] = number_of_affiliation_words
    dictionary_feature['number_of_analytical_words'] = number_of_analytical_words
    dictionary_feature['number_of_bad_words'] = number_of_bad_words
    dictionary_feature['number_of_certainty_words'] = number_of_certainty_words
    dictionary_feature['number_of_comparison_words'] = number_of_comparison_words
    dictionary_feature['number_of_common_words'] = number_of_common_words
    dictionary_feature['number_of_concern_words'] = number_of_concern_words
    dictionary_feature['number_of_emotional_tone_words'] = number_of_emotional_tone_words
    dictionary_feature['number_of_emotion_words'] = number_of_emotion_words
    dictionary_feature['number_of_interrogative_words'] = number_of_interrogative_words
    dictionary_feature['number_of_negative_words'] = number_of_negative_words
    dictionary_feature['number_of_power_words'] = number_of_power_words
    dictionary_feature['number_of_quantity_words'] = number_of_quantity_words
    dictionary_feature['number_of_reward_words'] = number_of_reward_words
    dictionary_feature['number_of_risk_words'] = number_of_risk_words
    dictionary_feature['number_of_slang_words'] = number_of_slang_words
    dictionary_feature['number_of_negative_words_Hu_Liu'] = number_of_negative_words_Hu_Liu
    dictionary_feature['number_of_positive_words_Hu_Liu'] = number_of_positive_words_Hu_Liu
    dictionary_feature['AFINN_word_score'] = AFINN_word_score

    return dictionary_feature


def main():
    with open('../newsfiles/sample.csv', 'r') as textfile:
        textcontent = textfile.read()
        print(dictionary_features(textcontent))


if __name__== "__main__":
    main()
