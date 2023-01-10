from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()


class PreProcess:
    text_key = "text"

    def __init__(self, df, text_key="text"):
        self.text_key = text_key
        self.df = df
        # lower case and remove NAs
        self.df[self.text_key] = df[self.text_key].dropna().str.lower()

        self.df[self.text_key] = df[self.text_key].str.replace("[^a-zA-Z ]", "", regex=True)

        # re-encode strings
        for i in range(len(self.df)):
            self.df.loc[i, self.text_key] = self.df.loc[i, self.text_key].encode("ascii", "ignore").decode("ascii")

    def lemmatize(self):
        for i in range(len(self.df)):
            # join array of lemmatized words in sentence
            lemmatized = " ".join([lemmatizer.lemmatize(word) for word in self.df[self.text_key].iloc[i].split(" ")])
            # update df
            self.df.loc[i, self.text_key] = lemmatized

    def returnDf(self):
        return self.df


if __name__ == "__main__":
    raise NotImplementedError("This file is not meant to run directly; its methods/class should be imported")
