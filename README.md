# Projeto ML Classifier N.L.P Noticiciario BBC News Classificacao de texto

__Bussines Problem:__
> Nos dias atuais possuímos uma grande frequência de informações disponíveis em sites, jornais e mídias de comunicação, estar a par destas notícias pode nos gerar insights de tendências de mercado, o grande ponto é que como seres humanos possuímos limitações para absorção e leitura de muitas notícias, sendo necessário a automatização dessas tarefas com a implementação de algoritmos de inteligência artificial para NLP do inglês natural language processing para a leitura automatizada e posterior classificação de um grande volume de notícias.

__Objetivo:__   
> Desenvolver um modelo de machine learning para o processamento de NLP, para notícias extraídas do site bbc (British Broadcasting Corporation) e classificação das mesmas em classes denominadas, bussines, entertainment, politics, sport,tech

__Autor:__  
   - Bruno Priantti.

__Apoio:__
- D. Greene and P. Cunningham. "Practical Solutions to the Problem of Diagonal Dominance in Kernel Document Clustering", Proc. ICML 2006.
    
__Contato:__  
  - bpriantti@gmail.com

__Encontre-me:__  
   -  https://www.linkedin.com/in/bpriantti/  
   -  https://github.com/bpriantti
   -  https://www.instagram.com/brunopriantti/
   
__Frameworks Utilizados:__

- Numpy: https://numpy.org/doc/  
- Pandas: https://pandas.pydata.org/
- Matplotlib: https://matplotlib.org/ 
- Seaborn: https://seaborn.pydata.org/  
- Plotly: https://plotly.com/  
- Scikit learn: https://scikit-learn.org/stable/index.html
- Statsmodels: https://www.statsmodels.org/stable/index.html

__Project Steps:__

__Step 01:__ Processo de Carregamento da base dados:

> Para projetos de processamento de nlp, torna-se necessário o pré-processamento dos textos a base em questão compila notícias do site bbc news nos períodos de 2005 a 2006, esta base é disponibilizada gratuitamente no site (link - abaixo), esta pesquisa foi fruto de um projeto de pesquisa em que os pesquisadores leram todas as notícias deste período documentaram e classificaram nas categorias: business, entertainment, politics, sport,tech

- http://mlg.ucd.ie/datasets/bbc.html

Fontes:
- https://www.bbc.com/news

__Step 02:__ Seraparando base em variavel dependente X e independente y.

> Realizou-se a separacao da base em features e alvos ao todo tem-se um total de 2225 noticias catalogadas na base.

      print(len(X)):
      2225
  
__Step 03:__ Adiante realizou-se o download das stop words, defini-se stop words como:

> Stop words (ou palavras de parada – tradução livre) são palavras que podem ser consideradas irrelevantes para o conjunto de resultados a ser exibido em uma busca realizada em uma search engine. Exemplos: as, e, os, de, para, com, sem, foi.

mais definicoes em:

https://pythonalgos.com/nlp-stop-words-when-and-why-to-use-them/

> para isso utilizou-se o framework NLTK, mais informacoes disponiveis em:

https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/index.xml

> como a base de noticias esta em ingles utilizaram-se as stopwords em ingles com o comando:

    my_stop_words = set(stopwords.words('english'))

__Step 04:__ Data Split, treinamento e teste:

> Em seguida realizou-se a separacao em dados de treinamento e teste para o modelo, utilizou-se uma proporcao de 70/30, com o comando disponivel pelo pacote sklearn, train data split:

    # Divisão em treino e teste (70/30)
    X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size = 0.30, random_state = 75)

__Step 05:__ Vetorizacao da base:

> O processo de vetorizacao pode ser definido como a transicricao de textos para numeros torna-se necessario a realizcao dessa etapa pois o processamento cumputacional funciona com base numerica e nao textual, foi utilizado o script abaixo para a vetorizacao dos textos:

    # Vet
    vectorizer = TfidfVectorizer(norm = None, stop_words = my_stop_words, max_features = 1000, decode_error = "ignore")

    # Observe que treinamos e aplicamos em treino e apenas aplicamos em teste!
    X_treino_vectors = vectorizer.fit_transform(X_treino)
    X_teste_vectors = vectorizer.transform(X_teste)
    
__Step 06:__ Treinamento dos modelos de Machine Learning:

> Adiante realizou-se o processo de treinamento dos modelos de  machine learning, optou-se por escolher os modelos de Regressao Logistica, multinomial, Random Forrest e Multinomial Nayve Baynes, optou-se por utilziar a tecnica de enssemble voting classifier, que nos permite unir as respostas dos modelos e ter uma agregacao de informacao contruibuindo para reduzir um erro e melhorar a acuracia e estabilidade das respostas do modelo, utilizou-se o framework scikit-learn para o desenvolvimento.

      # Modelos:

      # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
      mod1 = LogisticRegression(multi_class = 'multinomial', solver = 'lbfgs', random_state = 30, max_iter = 1000)

      # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
      mod2 = RandomForestClassifier(n_estimators = 1000, max_depth = 100, random_state = 1)

      # https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html
      mod3 = MultinomialNB()

      # Lista para o resultado
      resultado = []

      # Iniciando o modelo de votação
      # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html
      # https://scikit-learn.org/stable/modules/ensemble.html#voting-classifier
      voting_model = VotingClassifier(estimators = [ ('lg', mod1), ('rf', mod2), ('nb', mod3) ], voting = 'soft')

      print("\nModelo de Votação:\n")
      print(voting_model)
      
      # Treinamento
      voting_model = voting_model.fit(X_treino_vectors, y_treino)
    

__Step 07:__ Predict e Avaliando a Acuracia do Modelo:

> Realizou-se as etapas de predict para a base de teste e em seguida avaliacou-se a resposta da matriz de confusao e classificatio nrepport e acuracia do modelo, utilizando os scripts abaixo:

      # Previsões com dados de teste
      previsoes =  voting_model.predict(X_teste_vectors)

      # Grava o resultado
      resultado.append(accuracy_score(y_teste, previsoes))

      # Print
      print('\nAcurácia do Modelo:', accuracy_score(y_teste, previsoes),'\n')
      print('\n')
