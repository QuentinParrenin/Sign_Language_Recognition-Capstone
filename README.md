    Summarized overview 05/09/2023: 
    
Our purpose is to recognize fingerspelled addresses, phone numbers, and URLs from American Sign Language. The landmark of the hand position is already given, as well as the pose of the person with XYZ extracted with mediapipe. The speed is an important factor. The fingerspelling would be around 5+ letters per second.

About the data itself: 
There is a total of about 10 060 000 frames with 1630 total landmarks (sequence_id, frame, 468*3 face, 21*3 left hand, 21*3 right hand, 33*3 pose 'where 3 is xyz dimensions' / probably could drop all of the face landmarks, for alphabet they don't use facial or lip movement from what i've seen, this way we would have 63*2 + 99 total landmarks)
The average number of Frame per clip is 160. It is not certain yet but the current assumption is that videos have been filmed either in 24 or 30fps.
The train and test dataset are numbers, adresse and url. Meaning there will be quite of specific characters. (it is for those cases that real ASL speaker use the alphabet most of the time / otherwise they use words sign language)
The train landmark is divided in 68 parquet files. Each of ~1.5GB for a total of 100GB. Each file of about 150 000 frames and the 1630 landmarks. However we won't need the majority of those landmarks as saw in the first point. With only about 226 features the memory will be only a portion.

The current project is divided in 5 sections : 
1. ASL_EDA
2. ASL_Visuals
3. ASL_Dictionary_frequency_analysis
4. ASL_preprocessing
5. Research
6. Codes and Submission
7. Streamlit

1 ASL_EDA

from 160 000 frames (portion of the data)
Percentage of missing left hand values: 90.10
Percentage of missing right hand values: 50.50%
Percentage of rows with missing values for both left and right hand values for an identical frame : 41.02%
The pose landmark seem to be calibrated differently than hand : they have more negative values, and values higher than 1
There is exactly 1000 clip on the file for a total of 67 208 sequences.
The average number of frame is 160 per clip. The maximum being 488. (based on the maximum number of frame from each sequence_id +1 as we start from frame 0 ; the mean of the column would be unappropriate in this case)

2. ASL_Visual

- Got a visual of the video from our frames with landmark. It takes around 25 sec to have the visual of a sequence clip. (could be way lower if we don't take all face landmark but time is ok).
- Signers sign to not sign the space ' ' though it correspond to 5% of our labels character. 
- When signing numbers, for ex 654-464-1658, they do sign the '-' with a simple mouvement from their left to right. 
- Looked at a couple url visuals, there is a lot of missing data and the lips of the signer don't move what reconfort our hypothesis that we don't need those landmark for alphabet ASL. 
- In the sequence with an adresse including the word 'road', the word 'road' has indeed been signed in alphabet and not the word associated!
- Those visuals have been made pre-cleaning, cleaning the na will be important (specially every na starting or ending a sequence ; let's see next how to deal the best with within signing process NA values)

3. ASL_Dictionary_frequency_analysis3

Part1:
- 31.7% of the labels are composed of a number followed with a space without being a phone number (21 344 out of 67 208)
- 24.3% of the labels are composed of ``only numbers`` (including + - )  (16 356 out of 67 208)
- 16.2% of the labels are composed of www, http, .com, .net (10 886 out of 67 208)
- 12.4% of the labels are composed of ``only letters``. (8 347 out of 67 208)
- 84.6% of the labels classified with those 5 criteria.
- 
Part2:
- A dictionary of the frequency has been made. With the answer of how many times do any character is present in the file. Helping us to understand some exceptions case, and what letters to focus on for recognition as well as potential balance improvement.

Part3:
- Visualisation of some of the rare cases with symbols such as '$', '[', '('... As special character usually have movement in recognition, they are specially harder to detect with significant missing data.

4. ASL_preprocessing4
   
-Some feature reduction (The pose has 32 features, we clearly won't need them all as all their purpose is to have some potential information from the arm movement and have an idea of the distance of the body, therefore we will only keep arm, body and 1 point of the nose as center of the face/ We will keep 15 out of 32 for now.)
- Cleaning (some sequence really make no sense and there is a mistake from the data collection wisth high number of 6 frames, we will remove every sequence that has only 10 or less hand position information -> could remove more of them as 97.5% of our lables have more than 10 characters, and we could estimate that we need at least 2 to 4 landmark data for a character to have a decent prediction)
- Possibility of data augmentation (with a different scale, shift and degree)
- Filling with 0 (Works well in CNN and Transformers, doesn't disrupt the continuity of the sequence, have to make sure to do it after scaling)
- Padding (Meaning having some relatively identicla number of frames for the models to make some comparaison in an easier way)
  
5. Research5

`Have some some further research from papers and kaggle code/discussion. Found some interesting and useful process to get inspiration from to improve the process with efficiency and debugging.`

-For standardization, definition of a maximum length and automatically padd with 0 if the sequence has less frames.
[- (Too advanced for our deadlline on tf as only released on torch, but the concept of flash attention 2 on transformers from the paper released last month (stanford) has huge effect on memory and runtime. Using the asymmetric GPU memory hierarchy to bring significant memory saving (linear instead of quadratic), and runtime 2 to 4 times faster compared with optimized baseline. `flash attention 2 being specially impactful for long sequences`)]
- Having the mean and the standard deviation of our hand position, as well as the decision of the dominant hand ! 
- About 5% of the data character are spaces, though it seems like there is no sign character for it
- On this link, the mean, std of hand ; as well as Xtrain, Xval, ytrain, yval... are already shared as input https://www.kaggle.com/code/m4nugnzl/aslfr-eda-preprocessing-dataset-for-beginners/output 
- the end model config from the project having 0.675 - 4 About 5% of the data character are spaces, though it seems like there is no sign character for it ; chose GeLU activation function
- embedding with keras (hand landmark and frames)
- Awesome explaination of the transformer process with its code 
- encoder and decoder architecture and code
- loss function chose was categorical_crossentropy loss function with label smoothing support

6. Code and submission

The model has been finished just in time to try an attempt to submit on kaggle. The model worked though had to abandon some efficiency to fit the requirements of the competition on time. Being having a model under 40MB and an inference time inferior to 0.6 sec per prediction in average. 

7. Streamlit

The web app has been done on streamlit, with videos prerecorded and the label corresponding. The next step would be to have a direct result on the webcam. It implies extracting the data from the webcam in real time, passing it throw preprocessing and throw the trained model with inference. 

The current stage on date of (27/07)
    Includes 3 documents : 

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


Current Process:
The Pre-LN has been made and have decent result though should be improved in the futur.
The next step is the deplyment to have a real time recognition using streamlit, currently half way on it and have a streamlit functional for non-real time prediction. 

About the process it would be to start a first version of the model, find tricks for the quality of the data/time optimisation, augmentation if needed,build 3D CNN and transformer mvp, optimization and tuning, reiteration, and then post training quantization. And submit it to kaggle for feedback result and optimize.

ps the evaluation metric is the levenshtein metric (difference of characters between our output and the result ex : elephant , elephant = distance 0 ; zlephanw , elephant = distance 2 ; qwqwqwqw , elephant = distance 8) where score is (N-D)/N. Where N is total number of character and D total Levenshtein distance. The current top leaderboard score is around 0.78.
