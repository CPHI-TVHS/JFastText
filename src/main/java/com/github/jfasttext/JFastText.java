package com.github.jfasttext;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

import org.bytedeco.javacpp.PointerPointer;

public class JFastText {

	public static class ProbLabel {
		public String label;
		public float logProb;

		public ProbLabel(float logProb, String label) {
			this.logProb = logProb;
			this.label = label;
		}

		@Override
		public String toString() {
			return String.format("logProb = %f, label = %s", this.logProb, this.label);
		}
	}

	public static void main(String[] args) {
		JFastText jft = new JFastText();
		jft.runCmd(args);
	}

	private static List<String> stringVec2Strings(FastTextWrapper.StringVector sv) {
		List<String> strings = new ArrayList<>();
		for (int i = 0; i < sv.size(); i++) {
			strings.add(sv	.get(i)
							.getString());
		}
		return strings;
	}

	private FastTextWrapper.FastTextApi fta;

	public JFastText() {
		this.fta = new FastTextWrapper.FastTextApi();
	}

	public int getBucket() {
		return this.fta.getBucket();
	}

	public int getContextWindowSize() {
		return this.fta.getContextWindowSize();
	}

	public int getDim() {
		return this.fta.getDim();
	}

	public int getEpoch() {
		return this.fta.getEpoch();
	}

	public String getLabelPrefix() {
		return this.fta	.getLabelPrefix()
						.getString();
	}

	public List<String> getLabels() {
		return JFastText.stringVec2Strings(this.fta.getWords());
	}

	public String getLossName() {
		return this.fta	.getLossName()
						.getString();
	}

	public double getLr() {
		return this.fta.getLr();
	}

	public int getLrUpdateRate() {
		return this.fta.getLrUpdateRate();
	}

	public int getMaxn() {
		return this.fta.getMaxn();
	}

	public int getMinCount() {
		return this.fta.getMinCount();
	}

	public int getMinCountLabel() {
		return this.fta.getMinCountLabel();
	}

	public int getMinn() {
		return this.fta.getMinn();
	}

	public String getModelName() {
		return this.fta	.getModelName()
						.getString();
	}

	public int getNLabels() {
		return this.fta.getNLabels();
	}

	public int getNSampledNegatives() {
		return this.fta.getNSampledNegatives();
	}

	public int getNWords() {
		return this.fta.getNWords();
	}

	public String getPretrainedVectorsFileName() {
		return this.fta	.getPretrainedVectorsFileName()
						.getString();
	}

	public double getSamplingThreshold() {
		return this.fta.getSamplingThreshold();
	}

	public List<Float> getVector(String word) {
		FastTextWrapper.RealVector rv = this.fta.getVector(word);
		List<Float> wordVec = new ArrayList<>();
		for (int i = 0; i < rv.size(); i++) {
			wordVec.add(rv.get(i));
		}
		return wordVec;
	}

	public int getWordNgrams() {
		return this.fta.getWordNgrams();
	}

	public List<String> getWords() {
		return JFastText.stringVec2Strings(this.fta.getWords());
	}

	public void loadModel(String modelFile) {
		if (!new File(modelFile).exists()) {
			throw new IllegalArgumentException("Model file doesn't exist!");
		}
		if (!this.fta.checkModel(modelFile)) {
			throw new IllegalArgumentException("Model file's format is not compatible with this JFastText version!");
		}
		this.fta.loadModel(modelFile);
	}

	public String predict(String text) {
		List<String> predictions = this.predict(text, 1);
		return predictions.size() > 0 ? predictions.get(0) : null;
	}

	public List<String> predict(String text, int k) {
		if (k <= 0) {
			throw new IllegalArgumentException("k must be positive");
		}
		FastTextWrapper.StringVector sv = this.fta.predict(text, k);
		List<String> predictions = new ArrayList<>();
		for (int i = 0; i < sv.size(); i++) {
			predictions.add(sv	.get(i)
								.getString());
		}
		return predictions;
	}

	public ProbLabel predictProba(String text) {
		List<ProbLabel> probaPredictions = this.predictProba(text, 1);
		return probaPredictions.size() > 0 ? probaPredictions.get(0) : null;
	}

	public List<ProbLabel> predictProba(String text, int k) {
		if (k <= 0) {
			throw new IllegalArgumentException("k must be positive");
		}
		FastTextWrapper.FloatStringPairVector fspv = this.fta.predictProba(text, k);
		List<ProbLabel> probaPredictions = new ArrayList<>();
		for (int i = 0; i < fspv.size(); i++) {
			float logProb = fspv.first(i);
			String label = fspv	.second(i)
								.getString();
			probaPredictions.add(new ProbLabel(logProb, label));
		}
		return probaPredictions;
	}

	@SuppressWarnings("rawtypes")
	public void runCmd(String[] args) {
		// Prepend "fasttext" to the argument list so that it is compatible with C++'s main()
		String[] cArgs = new String[args.length + 1];
		cArgs[0] = "fasttext";
		System.arraycopy(args, 0, cArgs, 1, args.length);
		this.fta.runCmd(cArgs.length, new PointerPointer(cArgs));
	}

	public void test(String testFile) {
		this.test(testFile, 1);
	}

	public void test(String testFile, int k) {
		if (k <= 0) {
			throw new IllegalArgumentException("k must be positive");
		}
		this.fta.test(testFile, k);
	}

	public void unloadModel() {
		this.fta.unloadModel();
	}
}
