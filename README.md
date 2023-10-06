# Language-oriented-Semantic-Communiation
Code for "Language-oriented Communication with Semantic Coding and Knowledge Distillation for Text-to-Image Generation, Hyelin Nam, Jihong Park, Jinho Choi, Mehdi Bennis, Seong-Lyun Kim," submitted to ICASSP 2024.

# Settings for methods and importance
methods            |       importance          |     
-------------------|---------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------|
partBob            |                                   "most_attentive" Using language model from part of Bob's model to choose most influential or attentive word to transmit
partBob_clipTX     |                                 When Bob has CLIP language base. Same as above
allBob             |                                    "lowest LPIPS" Simulate with Bob's model and transmit the word with lowest LPIPS with the reference (desired) image
allBob_clipTX      |                                  When Bob has CLIP language base. Same as above
Alice              |                                     Using language model from part of Alice's model to choose most influential or attentive word to transmit
random             |                                  Randomly transmit words
bothAtt            |   contribute              |    Contribution is measured by the sum of word impact to the other words (sum row and col of a word_index in the attention matrix of language model), transmit the word that Alice and Bob has the most similar value of contribution (smallest difference of Alice's sum and Bob's sum) 
                   |     combination           |     Combination is measured by the value of word to another word (value of x_cor: word, y_cor: another word in the attention matrix of language model), transmit the word that Alice and Bob has the most similar value of combination with last sent word. First transmitting word is selected through 'contribution' method
bothcrossatt       |                                  Compare Alice and Bob's contribution map (which part of image each word is describing / generating), transmit the word with similar map (smallest difference of two maps)
prompttune         | nouns                     |   Transmit nouns first, in context-sequence
                   |      nounsverbs           |      Transmit nouns and verbs first, in context-sequence
                   |      heads                |        "SSC" Transmit heads first, in context-sequence
                   |      heads_partBob        |     Transmit heads first, in context like partBob (Measure priority within the head words through language model from Bob's model)
                   |      heads_random         |    Transmit heads first, in random-sequence
                   |      heads_skd            |       "SKD" Finetune Alice and transmit heads first, in context-sequence
      
