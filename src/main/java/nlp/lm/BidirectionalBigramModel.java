package nlp.lm;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/** 
 * @author Ray Mooney
 * A simple bigram language model that uses simple fixed-weight interpolation
 * with a unigram model for smoothing.
*/

enum ModelType {
    FORWARD,
    BACKWARD,
    BIDIRECTIONAL
}

public class BidirectionalBigramModel {

    /** Unigram model that maps a token to its unigram probability */
    public Map<String, DoubleValue> unigramMapForward = null;

    public Map<String, DoubleValue> unigramMapBackward = null;

    /**  Bigram model that maps a bigram as a string "A\nB" to the
     *   P(B | A) */
    public Map<String, DoubleValue> bigramMapForward = null;

    public Map<String, DoubleValue> bigramMapBackward = null;

    /** Total count of tokens in training data */
    public double tokenCount = 0;

    /** Maintain separate token counts for forward and backward parsing*/
    public double tokenCountForward = 0;
    public double tokenCountBackward = 0;

    /** Interpolation weight for unigram model */
    public double lambda1 = 0.1;

    /** Interpolation weight for bigram model */
    public double lambda2 = 0.9;

    /** Interpolation weight for forward model */
    public double lambda3 = 0.5;

    /** Interpolation weight for forward model */
    public double lambda4 = 0.5;

    /** Type of Bigram model */
    public ModelType modelType;

    /** Initialize model with empty hashmaps with initial
     *  unigram entries for sentence start (<S>), sentence end (</S>)
     *  and unknown tokens */
    public BidirectionalBigramModel(ModelType modelType) {
        unigramMapForward = new HashMap<String, DoubleValue>();
        unigramMapBackward = new HashMap<String, DoubleValue>();

		bigramMapForward = new HashMap<String, DoubleValue>();
		bigramMapBackward = new HashMap<String, DoubleValue>();

		unigramMapForward.put("<S>", new DoubleValue());
        unigramMapForward.put("</S>", new DoubleValue());
        unigramMapForward.put("<UNK>", new DoubleValue());

        unigramMapBackward.put("<S>", new DoubleValue());
        unigramMapBackward.put("</S>", new DoubleValue());
        unigramMapBackward.put("<UNK>", new DoubleValue());

        this.modelType = modelType;
    }

    /** Train the model on a List of sentences represented as
     *  Lists of String tokens */
    public void train (List<List<String>> sentences) {
        if (modelType == ModelType.FORWARD) {
            trainForward(sentences);
        } else if (modelType == ModelType.BACKWARD) {
            trainBackward(sentences);
        } else {
            trainForward(sentences);
            trainBackward(sentences);
        }
    }

    public void trainForward (List<List<String>> sentences) {
        // Accumulate unigram and bigram counts in maps
        trainSentencesForward(sentences);
        // Compure final unigram and bigram probs from counts
        calculateProbs(unigramMapForward, bigramMapForward, tokenCountForward);
    }

    public void trainBackward (List<List<String>> sentences) {
        // Accumulate unigram and bigram counts in maps
        trainSentencesBackward(sentences);
        // Compure final unigram and bigram probs from counts
        calculateProbs(unigramMapBackward, bigramMapBackward, tokenCountBackward);
    }

    /** Accumulate unigram and bigram counts for these sentences */
    public void trainSentencesForward (List<List<String>> sentences) {
        tokenCount = 0;
        for (List<String> sentence : sentences) {
            trainSentence(sentence, unigramMapForward, bigramMapForward, "<S>", "</S>");
        }
        tokenCountForward = tokenCount;
    }

    /** Accumulate unigram and bigram counts for these sentences in backward direction */
    public void trainSentencesBackward (List<List<String>> sentences) {
        tokenCount = 0;
        for (List<String> sentence : sentences) {
            List<String> reverseSentence = reverseSentence(sentence);
//            System.out.print(sentence + " " + reverseSentence);
            trainSentence(reverseSentence, unigramMapBackward, bigramMapBackward, "</S>", "<S>");
        }
        tokenCountBackward = tokenCount;
    }

    /** Accumulate unigram and bigram counts for this sentence */
    public void trainSentence (List<String> sentence, Map<String, DoubleValue> unigramMap, Map<String, DoubleValue> bigramMap, String startToken, String endToken) {
		// First count an initial start sentence token
		String prevToken = startToken;
		DoubleValue unigramValue = unigramMap.get(startToken);
		unigramValue.increment();
		tokenCount++;
		// For each token in sentence, accumulate a unigram and bigram count
		for (String token : sentence) {
			unigramValue = unigramMap.get(token);
			// If this is the first time token is seen then count it
			// as an unkown token (<UNK>) to handle out-of-vocabulary
			// items in testing
			if (unigramValue == null) {
				// Store token in unigram map with 0 count to indicate that
				// token has been seen but not counted
				unigramMap.put(token, new DoubleValue());
				token = "<UNK>";
				unigramValue = unigramMap.get(token);
			}
			unigramValue.increment();    // Count unigram
			tokenCount++;               // Count token
			// Make bigram string
			String bigram = bigram(prevToken, token);
			DoubleValue bigramValue = bigramMap.get(bigram);
			if (bigramValue == null) {
				// If previously unseen bigram, then
				// initialize it with a value
				bigramValue = new DoubleValue();
				bigramMap.put(bigram, bigramValue);
			}
			// Count bigram
			bigramValue.increment();
			prevToken = token;
		}
		// Account for end of sentence unigram
		unigramValue = unigramMap.get(endToken);
		unigramValue.increment();
		tokenCount++;
		// Account for end of sentence bigram
		String bigram = bigram(prevToken, endToken);
		DoubleValue bigramValue = bigramMap.get(bigram);
		if (bigramValue == null) {
			bigramValue = new DoubleValue();
			bigramMap.put(bigram, bigramValue);
		}
		bigramValue.increment();
    }

    /** Compute unigram and bigram probabilities from unigram and bigram counts */
    public void calculateProbs(Map<String, DoubleValue> unigramMap, Map<String, DoubleValue> bigramMap, double numberOfTokens) {
		// Set bigram values to conditional probability of second token given first
		for (Map.Entry<String, DoubleValue> entry : bigramMap.entrySet()) {
			// An entry in the HashMap maps a token to a DoubleValue
			String bigram = entry.getKey();
			// The value for the token is in the value of the DoubleValue
			DoubleValue value = entry.getValue();
			double bigramCount = value.getValue();
			String token1 = bigramToken1(bigram); // Get first token of bigram
			// Prob is ratio of bigram count to token1 unigram count
			double condProb = bigramCount / unigramMap.get(token1).getValue();
			// Set map value to conditional probability
			value.setValue(condProb);
		}
		// Store unigrams with zero count to remove from map
		List<String> zeroTokens = new ArrayList<String>();
		// Set unigram values to unigram probability
		for (Map.Entry<String, DoubleValue> entry : unigramMap.entrySet()) {
			// An entry in the HashMap maps a token to a DoubleValue
			String token = entry.getKey();
			// Uniggram count is the current map value
			DoubleValue value = entry.getValue();
			double count = value.getValue();
			if (count == 0) {
                // If count is zero (due to first encounter as <UNK>)
                // then remove save it to remove from map
                zeroTokens.add(token);
            } else {
                // Set map value to prob of unigram
                value.setValue(count / numberOfTokens);
            }
		}
		// Remove zero count unigrams from map
		for (String token : zeroTokens) {
			unigramMap.remove(token);
		}
    }

    /** Return bigram string as two tokens separated by a newline */
    public String bigram (String prevToken, String token) {
		return prevToken + "\n" + token;
    }

    /** Return fist token of bigram (substring before newline) */
    public String bigramToken1 (String bigram) {
	int newlinePos = bigram.indexOf("\n");
	return bigram.substring(0,newlinePos);
    }

    /** Return second token of bigram (substring after newline) */
    public String bigramToken2 (String bigram) {
	int newlinePos = bigram.indexOf("\n");
	return bigram.substring(newlinePos + 1, bigram.length());
    }

    /** Print model as lists of unigram and bigram probabilities */
    public void print(Map<String, DoubleValue> unigramMap, Map<String, DoubleValue> bigramMap) {
        System.out.println("Unigram probs:");
        for (Map.Entry<String, DoubleValue> entry : unigramMap.entrySet()) {
            // An entry in the HashMap maps a token to a DoubleValue
            String token = entry.getKey();
            // The value for the token is in the value of the DoubleValue
            DoubleValue value = entry.getValue();
            System.out.println(token + " : " + value.getValue());
        }
        System.out.println("\nBigram probs:");
        for (Map.Entry<String, DoubleValue> entry : bigramMap.entrySet()) {
            // An entry in the HashMap maps a token to a DoubleValue
            String bigram = entry.getKey();
            // The value for the token is in the value of the DoubleValue
            DoubleValue value = entry.getValue();
            System.out.println(bigramToken2(bigram) + " given " + bigramToken1(bigram) +
                       " : " + value.getValue());
        }
    }

    /** Use sentences as a test set to evaluate the model. Print out perplexity
     *  of the model for this test data */
    public void test (List<List<String>> sentences) {
        if (modelType == ModelType.BIDIRECTIONAL) {
            System.err.println("Cannot compute perplexity for Bidirectional model with prediction of start/end of sentence! Please compute word perplexity.");
        }
        // Compute log probability of sentence to avoid underflow
        double totalLogProb = 0;
        // Keep count of total number of tokens predicted
        double totalNumTokens = 0;
        // Accumulate log prob of all test sentences
        for (List<String> sentence : sentences) {
            // Num of tokens in sentence plus 1 for predicting </S>
            totalNumTokens += sentence.size() + 1;
            // Compute log prob of sentence
            double sentenceLogProb = 0.0;
            if (modelType == ModelType.FORWARD) {
                sentenceLogProb = sentenceLogProb(sentence, unigramMapForward, bigramMapForward, "<S>", "</S>");
            } else if (modelType == ModelType.BACKWARD) {
                sentenceLogProb = sentenceLogProb(reverseSentence(sentence), unigramMapBackward, bigramMapBackward, "</S>", "<S>");
            }
            //	    System.out.println(sentenceLogProb + " : " + sentence);
            // Add to total log prob (since add logs to multiply probs)
            totalLogProb += sentenceLogProb;
        }
        // Given log prob compute perplexity
        double perplexity = Math.exp(-totalLogProb / totalNumTokens);
        System.out.println("Perplexity = " + perplexity );
    }

    List<String> reverseSentence(List<String> sentence) {
        List<String> reverseSentence = new ArrayList<String>();
        for (int index = sentence.size()-1; index >= 0; index--) {
            reverseSentence.add(sentence.get(index));
        }
        return reverseSentence;
    }

    /* Compute log probability of sentence given current model */
    public double sentenceLogProb (List<String> sentence, Map<String, DoubleValue> unigramMap, Map<String, DoubleValue> bigramMap, String startToken, String endToken) {
        // Set start-sentence as initial token
        String prevToken = startToken;
        // Maintain total sentence prob as sum of individual token
        // log probs (since adding logs is same as multiplying probs)
        double sentenceLogProb = 0;
        // Check prediction of each token in sentence
        for (String token : sentence) {
            // Retrieve unigram prob
            DoubleValue unigramVal = unigramMap.get(token);
            if (unigramVal == null) {
                // If token not in unigram model, treat as <UNK> token
                token = "<UNK>";
                unigramVal = unigramMap.get(token);
            }
            // Get bigram prob
            String bigram = bigram(prevToken, token);
            DoubleValue bigramVal = bigramMap.get(bigram);
            // Compute log prob of token using interpolated prob of unigram and bigram
            double logProb = Math.log(interpolatedProb(unigramVal, bigramVal));
            // Add token log prob to sentence log prob
            sentenceLogProb += logProb;
            // update previous token and move to next token
            prevToken = token;
        }
        // Check prediction of end of sentence token
        DoubleValue unigramVal = unigramMap.get(endToken);
        String bigram = bigram(prevToken, endToken);
        DoubleValue bigramVal = bigramMap.get(bigram);
        double logProb = Math.log(interpolatedProb(unigramVal, bigramVal));
        // Update sentence log prob based on prediction of </S>
        sentenceLogProb += logProb;
        return sentenceLogProb;
    }

    /** Like test1 but excludes predicting end-of-sentence when computing perplexity */
    public void test2 (List<List<String>> sentences) {
        double totalLogProb = 0;
        double totalNumTokens = 0;
        for (List<String> sentence : sentences) {
            totalNumTokens += sentence.size();
            double sentenceLogProb = 0.0;
            if (modelType == ModelType.FORWARD) {
                sentenceLogProb = sentenceLogProb2(sentence, unigramMapForward, bigramMapForward, "<S>", "</S>");
            } else if (modelType == ModelType.BACKWARD) {
                sentenceLogProb = sentenceLogProb2(reverseSentence(sentence), unigramMapBackward, bigramMapBackward, "</S>", "<S>");
            } else if (modelType == ModelType.BIDIRECTIONAL) {
                sentenceLogProb = sentenceLogProbBidirectional(sentence);
            }

            //	    System.out.println(sentenceLogProb + " : " + sentence);
            totalLogProb += sentenceLogProb;
        }
        double perplexity = Math.exp(-totalLogProb / totalNumTokens);
        System.out.println("Word Perplexity = " + perplexity );
    }

    /** Like sentenceLogProb but excludes predicting end-of-sentence when computing prob */
    public double sentenceLogProbBidirectional (List<String> sentence) {
        double forwardProbs[] = sentenceTokenProbs2(sentence, unigramMapForward, bigramMapForward, "<S>");
        double backwardProbs[] = sentenceTokenProbs2(reverseSentence(sentence), unigramMapBackward, bigramMapBackward, "</S>");
        assert (forwardProbs.length == backwardProbs.length);

        double sentenceLogProb = 0.0;
        for (int index = 0; index < forwardProbs.length; index++) {
            double logProb = Math.log(lambda3 * forwardProbs[index] + lambda4 * backwardProbs[index]);
            sentenceLogProb += logProb;
        }

        return sentenceLogProb;
    }

    /** Like sentenceLogProb but excludes predicting end-of-sentence when computing prob */
    public double sentenceLogProb2 (List<String> sentence, Map<String, DoubleValue> unigramMap, Map<String, DoubleValue> bigramMap, String startToken, String endToken) {
        String prevToken = "<S>";
        double sentenceLogProb = 0;
        for (String token : sentence) {
            DoubleValue unigramVal = unigramMap.get(token);
            if (unigramVal == null) {
                token = "<UNK>";
                unigramVal = unigramMap.get(token);
            }
            String bigram = bigram(prevToken, token);
            DoubleValue bigramVal = bigramMap.get(bigram);
            double logProb = Math.log(interpolatedProb(unigramVal, bigramVal));
            sentenceLogProb += logProb;
            prevToken = token;
        }
        return sentenceLogProb;
    }

    /** Returns vector of probabilities of predicting each token in the sentence
     *  including the end of sentence */
    public double[] sentenceTokenProbs (List<String> sentence, Map<String, DoubleValue> unigramMap, Map<String, DoubleValue> bigramMap, String startToken, String endToken) {
        // Set start-sentence as initial token
        String prevToken = startToken;
        // Vector for storing token prediction probs
        double[] tokenProbs = new double[sentence.size() + 1];
        // Token counter
        int i = 0;
        // Compute prob of predicting each token in sentence
        for (String token : sentence) {
            DoubleValue unigramVal = unigramMap.get(token);
            if (unigramVal == null) {
                token = "<UNK>";
                unigramVal = unigramMap.get(token);
            }
            String bigram = bigram(prevToken, token);
            DoubleValue bigramVal = bigramMap.get(bigram);
            // Store prediction prob for i'th token
            tokenProbs[i] = interpolatedProb(unigramVal, bigramVal);
            prevToken = token;
            i++;
        }
        // Check prediction of end of sentence
        DoubleValue unigramVal = unigramMap.get(endToken);
        String bigram = bigram(prevToken, endToken);
        DoubleValue bigramVal = bigramMap.get(bigram);
        // Store end of sentence prediction prob
        tokenProbs[i] = interpolatedProb(unigramVal, bigramVal);
        return tokenProbs;
    }

    /** Returns vector of probabilities of predicting each token in the sentence
     *  without the end of sentence */
    public double[] sentenceTokenProbs2 (List<String> sentence, Map<String, DoubleValue> unigramMap, Map<String, DoubleValue> bigramMap, String startToken) {
        // Set start-sentence as initial token
        String prevToken = startToken;
        // Vector for storing token prediction probs
        double[] tokenProbs = new double[sentence.size()];
        // Token counter
        int i = 0;
        // Compute prob of predicting each token in sentence
        for (String token : sentence) {
            DoubleValue unigramVal = unigramMap.get(token);
            if (unigramVal == null) {
                token = "<UNK>";
                unigramVal = unigramMap.get(token);
            }
            String bigram = bigram(prevToken, token);
            DoubleValue bigramVal = bigramMap.get(bigram);
            // Store prediction prob for i'th token
            tokenProbs[i] = interpolatedProb(unigramVal, bigramVal);
            prevToken = token;
            i++;
        }
        return tokenProbs;
    }

    /** Interpolate bigram prob using bigram and unigram model predictions */
    public double interpolatedProb(DoubleValue unigramVal, DoubleValue bigramVal) {
        double bigramProb = 0;
        // In bigram unknown then its prob is zero
        if (bigramVal != null)
            bigramProb = bigramVal.getValue();
        // Linearly combine weighted unigram and bigram probs
        return lambda1 * unigramVal.getValue() + lambda2 * bigramProb;
    }

    public static int wordCount (List<List<String>> sentences) {
        int wordCount = 0;
        for (List<String> sentence : sentences) {
            wordCount += sentence.size();
        }
        return wordCount;
    }

    /** Train and test a bigram model.
     *  Command format: "nlp.lm.BigramModel [DIR]* [TestFrac]" where DIR 
     *  is the name of a file or directory whose LDC POS Tagged files should be 
     *  used for input data; and TestFrac is the fraction of the sentences
     *  in this data that should be used for testing, the rest for training.
     *  0 < TestFrac < 1
     *  Uses the last fraction of the data for testing and the first part
     *  for training.
     */
    public static void main(String[] args) throws IOException {
		// All but last arg is a file/directory of LDC tagged input data
		File[] files = new File[args.length - 1];
		for (int i = 0; i < files.length; i++)
			files[i] = new File(args[i]);
		// Last arg is the TestFrac
		double testFraction = Double.valueOf(args[args.length -1]);
		// Get list of sentences from the LDC POS tagged input files
		List<List<String>> sentences = 	POSTaggedFile.convertToTokenLists(files);
//		System.out.print(sentences);
		int numSentences = sentences.size();
		// Compute number of test sentences based on TestFrac
		int numTest = (int)Math.round(numSentences * testFraction);
		// Take test sentences from end of data
		List<List<String>> testSentences = sentences.subList(numSentences - numTest, numSentences);
		// Take training sentences from start of data
		List<List<String>> trainSentences = sentences.subList(0, numSentences - numTest);
		System.out.println("# Train Sentences = " + trainSentences.size() +
				   " (# words = " + wordCount(trainSentences) +
				   ") \n# Test Sentences = " + testSentences.size() +
				   " (# words = " + wordCount(testSentences) + ")");
		System.out.println("--------------------------------------- FORWARD MODEL -------------------------------------");
		// Create a bigram model and train it.
		BidirectionalBigramModel forwardModel = new BidirectionalBigramModel(ModelType.FORWARD);
		System.out.println("Training...");
		forwardModel.train(trainSentences);
		// Test on training data using test and test2
		forwardModel.test(trainSentences);
		forwardModel.test2(trainSentences);
		System.out.println("Testing...");
		// Test on test data using test and test2
		forwardModel.test(testSentences);
		forwardModel.test2(testSentences);

        System.out.println("--------------------------------------- BACKWARD MODEL -------------------------------------");
        BidirectionalBigramModel backwardModel = new BidirectionalBigramModel(ModelType.BACKWARD);
        System.out.println("Training...");
        backwardModel.train(trainSentences);
        // Test on training data using test and test2
        backwardModel.test(trainSentences);
        backwardModel.test2(trainSentences);
        System.out.println("Testing...");
        // Test on test data using test and test2
        backwardModel.test(testSentences);
        backwardModel.test2(testSentences);

        System.out.println("--------------------------------------- BIDIRECTIONAL MODEL -------------------------------------");
        BidirectionalBigramModel bidirectionalModel = new BidirectionalBigramModel(ModelType.BIDIRECTIONAL);
        bidirectionalModel.lambda3 = 0.5;
        bidirectionalModel.lambda4 = 0.5;
        System.out.println("Training...");
        bidirectionalModel.train(trainSentences);
        // Test on training data using test and test2
//        bidirectionalModel.test(trainSentences);
        bidirectionalModel.test2(trainSentences);
        System.out.println("Testing...");
        // Test on test data using test and test2
//        bidirectionalModel.test(testSentences);
        bidirectionalModel.test2(testSentences);
//        System.out.println(bidirectionalModel.tokenCountForward + " " + bidirectionalModel.tokenCountBackward);
    }

}
