    summarized overview: 
Our purpose is to recognize fingerspelled adresse, phone numbers and url from American Sign language. The landmark of the hand position are already given as well as the pose of the person with xyz extracted with mediapipe. The speed is an important factor, the fingerspelling would be around 5+ letters per second.

About the data : 
There is a total of about 10 060 000 frames with 1630 total landmarks (sequence_id, frame, 468*3 face, 21*3 left hand, 21*3 right hand, 33*3 pose 'where 3 is xyz dimensions' / probably could drop all of the face landmark, for alphabet they don't use facial or lip movement from what i've seen, this way we would have 63*2 + 99 total landmarks)
The average number of Frame per clip is 160. It is not certain yet but the current assumption is that videos have been filmed either in 24 or 30fps.
The train and test dataset are numbers, adresse and url. Meaning there will be quite of specific characters. (it is for those cases that real ASL speaker use the alphabet most of the time / otherwise they use words sign language)
The train landmark is divided in 68 parquet files. Each of ~1.5GB for a total of 100GB. Each file of about 150 000 frames and the 1630 landmarks. However we won't need the majority of those landmarks as saw in the first point. With only about 226 features the memory will be only a portion.

The current stage (27/07)
    Includes 3 documents : 
    
  EDA
    - 50% of right land landmark are missing in a given frame
    - 90% of left hand landmark are missing in a given frame
    - 41% of both hand landmark are missing in a given frame

  Dictionary of patterns and frequences
    - Understanding of our main pattern in term of labels.
        -32% is a set of number followed by a space (adresses)
        -24.3% is composed of only numbers or special character such as + - (phone number)
        -16.2% is composed of www, http, .com, .net (url)
        -12.4% are composed of only letters (mainly names, ex : javier le)
        With those criteria 84.6% of the lables are classified. (Can go deeper on the analyzes to bring out some patterns though they become less and less clears)
        
      - Dictionanary of frequences
        -The frequency of every individual letter, number, and special character in the text.

    Visuals
      -Made some 2d visuals of the clips we have of the positions of the hand, pose and face.
      -Helpful to understand more the context of our data as we only have numbers provided, and a limited amount of information are given regarding the creation/collection of the data and its quality.

Aspects and specificity to think about: 
Words with double letter
most signer for a world such as ‘FOOD’ would sign F, O then slide their hand to the outside while holding the O letter on the hand.
An alternative to sliding double letters is to unflex the fingers and reform the letter.
A final method that signer use sometimes is to bounce the letter to indicate that it is a double letter.
Fingerspelling can be very fast, observed from 5 to 8 characters per second. (https://www.purdue.edu/tislr10/pdfs/Quinto-Pozos fingerspelling.pdf)
Words that are unknown to a conversation are slower (name, phone, mail, address). Known words are faster and have more elision.
Longer words are fingerspelled quicker than shorter words.
Just like phonemes in speech, some letters may be elided (dropped). For example for known words with context, some experienced signer might sign ‘elephant’ as ELE-muddle-NT for example.

We really have to give an extra focus to the context (ps: 1 some letter are really fast, 2 some letter are somewhat similar, 3 unknown words are slower reducing bad effects, 4 some words have missing letter on spelling, 5 reduction of error) more likely use NLP ?

Some specific letters distinctions :
Only two letters in ASL fingerspelling have motion: J and Z.
P and Q are distinct in their own way. For both, the hand is tilted down, and that only occurs for these two letters. the downward motion made when switching from a preceding letter to a P or Q followed by the upward motion for the next letter should be a pretty strong feature
Have to dig further on: 
Learn more about the special character of ASL. (as we deal with url they would be quite an important thing to distinguish approppriatly)
How to effectively import the data from parquet for it to be faster. (for now I used pd.read_parquet)
How speech recognition is done ; likely to have a lot of similarities in the process 
Seem like some people used pytorch and reconvert it later in tensorflow.lite, is it a method efficient enough ?
There should be the video on the parquet file, how to visualize it. Should watch a hundred to have some insights from it
post training quantization would probably be efficient to do - learn further about it
How much of the data should we use for the training and validation (out of the 68 parquet files, I don't think we need to make a test dataset as we could submit up to 5 times a day on the kaggle where we would have the automatic score from it)
Is there faster way to work on the project than my cpu/vsc without expenses ?
How do we prefer to communicate, do we use miro mainly for strategy or organization and exchange on more detail topic and ideas such this mail by mail or get to a googledoc or else
learn more on the trick and process from top solutions from the previous challenge on asl word recognition


Current idea of the process : 
The current idea of the model is to build a 3D CNN and transformers. 
About the process it would be to depthen the cleaning, find tricks for the quality of the data/time optimisation, augmentation if needed,build 3D CNN and transformer mvp, optimization and tuning, reiteration, and then post training quantization. And submit it to kaggle for feedback result and optimize.

ps the evaluation metric is the levenshtein metric (difference of characters between our output and the result ex : elephant , elephant = distance 0 ; zlephanw , elephant = distance 2 ; qwqwqwqw , elephant = distance 8) where score is (N-D)/N. Where N is total number of character and D total Levenshtein distance. The current top leaderboard score is around 0.78.
