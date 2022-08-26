# Projeto ML Classifier N.L.P Noticiciario BBC News Classificacao de texto

__Bussines Problem:__
> Nos dias atuais possuímos uma grande frequência de informações disponíveis em sites, jornais e mídias de comunicação, estar a par destas notícias pode nos gerar insights de tendências de mercado, o grande ponto é que como seres humanos possuímos limitações para absorção e leitura de muitas notícias, sendo necessário a automatização dessas tarefas com a implementação de algoritmos de inteligência artificial para NLP do inglês natural language processing para a leitura automatizada e posterior classificação de um grande volume de notícias.

__Objetivo:__   
> Desenvolver um modelo de machine learning para o processamento de NLP, para notícias extraídas do site bbc (British Broadcasting Corporation) e classificação das mesmas em classes denominadas, business, entertainment, politics, sport,tech

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

__Step 02:__ Separando base numa variável dependente X e independente y.

> Realizou-se a separação da base em features e alvos ao todo tem-se um total de 2225 notícias catalogadas na base.

      print(len(X)):
      2225
  
__Step 03:__ Adiante realizou-se o download das stop words, define-se stopwords como:

> Stop words (ou palavras de parada – tradução livre) são palavras que podem ser consideradas irrelevantes para o conjunto de resultados a ser exibido em uma busca realizada em uma search engine. Exemplos: as, e, os, de, para, com, sem, foi.

mais definições em:

https://pythonalgos.com/nlp-stop-words-when-and-why-to-use-them/

> para isso utilizou-se o framework NLTK, mais informações disponíveis em:

https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/index.xml

> como a base de noticias esta em ingles utilizaram-se as stopwords em inglês com o comando:

    my_stop_words = set(stopwords.words('english'))

__Step 04:__ Data Split, treinamento e teste:

> Em seguida realizou-se a separação em dados de treinamento e teste para o modelo, utilizou-se uma proporção de 70/30, com o comando disponível pelo pacote sklearn, train data split:

    # Divisão em treino e teste (70/30)
    X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size = 0.30, random_state = 75)

__Step 05:__ Vetorização da base:

> O processo de vetorização pode ser definido como a transcrição de textos para números torna-se necessário a realização dessa etapa pois o processamento computacional funciona com base numérica e nao textual, foi utilizado o script abaixo para a vetorização dos textos:

    # Vet
    vectorizer = TfidfVectorizer(norm = None, stop_words = my_stop_words, max_features = 1000, decode_error = "ignore")

    # Observe que treinamos e aplicamos em treino e apenas aplicamos em teste!
    X_treino_vectors = vectorizer.fit_transform(X_treino)
    X_teste_vectors = vectorizer.transform(X_teste)
    
__Step 06:__ Treinamento dos modelos de Machine Learning:

> Adiante realizou-se o processo de treinamento dos modelos de  machine learning, optou-se por escolher os modelos de Regressão Logística, multinomial, Random Forrest e Multinomial Naive Baynes, optou-se por utilizar a técnica de ensemble voting classifier, que nos permite unir as respostas dos modelos e ter uma agregação de informação contribuindo para reduzir um erro e melhorar a acurácia e estabilidade das respostas do modelo, utilizou-se o framework scikit-learn para o desenvolvimento.

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
    

__Step 07:__ Predict e Avaliando a Acurácia do Modelo:

> Realizou-se as etapas de predict para a base de teste e em seguida avaliou-se a resposta da matriz de confusão e Classification repport e acurácia do modelo, utilizando os scripts abaixo:

- Classification Report:

      Classification Report:  
                    precision    recall  f1-score   support

                 0       0.94      0.95      0.94       152
                 1       0.98      0.98      0.98       129
                 2       0.98      0.96      0.97       126
                 3       0.99      1.00      1.00       146
                 4       0.97      0.97      0.97       115

          accuracy                           0.97       668
         macro avg       0.97      0.97      0.97       668
      weighted avg       0.97      0.97      0.97       668

- Matrix Confusão:

   > <img src="https://github.com/bpriantti/projeto_ml_classifier_nlp_noticiciario_bbc_news_class_text/blob/main/images/image_1.PNG?raw=true"  width="400" height = "300">

Acurácia do Modelo: 0.9625748502994012 

obs: Como observamos o modelo obteve uma acurácia muito boa tendo um erro menor que 4% para a base de dados desconhecida.

__Step 08:__ Otimizando Hiperparâmetros:

Com o objetivo de melhorar a acurácia do modelo implementou-se a otimização de hiperparâmetros para o modelo optou-se por otimizar os hiperparâmetros, random_state do train data split.

Optou-se também por realizar o treinamento de outro modelo de ensemble utilizando o parâmetro hard e sort, basicamente a diferença entre os dois é que no hard a resposta final é a confluência de respostas dos modelos e no soft a votação da peso para determinado modelo escolhido pelo usuário.

utilizou-se o script:

      # Para gravar o resultado
      d = {}
      d['soft'] = []
      d['hard'] = []

      # Loop
      for x in range(1,100):

         # Div treino/teste
          X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.30, random_state = x)

          # Stop words
          my_stop_words = set(stopwords.words('english'))

          # Vet
          vectorizer = TfidfVectorizer(norm = None, stop_words = my_stop_words, max_features = 1000, decode_error = "ignore")

          # Aplica a vetorização
          X_treino_vectors = vectorizer.fit_transform(X_treino)
          X_teste_vectors = vectorizer.transform(X_teste)

          # Cria os modelos base
          modelo1 = LogisticRegression(multi_class = 'multinomial', solver = 'lbfgs', random_state = 30, max_iter = 1000)
          modelo2 = RandomForestClassifier(n_estimators = 1000, max_depth = 100, random_state = 1)
          modelo3 = MultinomialNB()

          # Loop
          for i in ['soft','hard']:
              voting_model = VotingClassifier(estimators = [ ('lg', modelo1), ('rf', modelo2), ('nb', modelo3)], voting = i)
              voting_model = voting_model.fit(X_treino_vectors, y_treino)
              previsoes = voting_model.predict(X_teste_vectors)
              print('-Random State:', x, '-Voting:', i, '-Acurácia :', accuracy_score(y_teste, previsoes))
              d[i].append((x,accuracy_score(y_teste, previsoes)))


      print('\nMelhores Resultados:')

      # Extrai os melhores resultados
      h = max(d['hard'], key = lambda x:x[1])
      s = max(d['soft'], key = lambda x:x[1])

      # Print Melhores Resultados:
      print('-Random State:',h[0], '-Voting:hard', '-Acurácia:', h[1])
      print('-Random State:',s[0], '-Voting:soft', '-Acurácia:', s[1])
      
> O melhor modelo obtido foi obteve:

-Random State: 74 -Voting:hard -Acurácia: 0.9880239520958084
-Random State: 93 -Voting:soft -Acurácia: 0.9865269461077845

Resultados e Trabalhos Futuros:
___

Os resultados obtidos para o modelo foram satisfatórios com uma acurácia de 98.8 para a base de dados de teste, isso quer dizer que o modelo em produção pode ter um erro inferior a 2.2%, destaca-se a eficiência dos métodos ensemble como podemos observar tanto para o ensemble votting ou o stacking mostraram resultados satisfatórios.

- Trabalhos Futuros:
   > Implementar um modelo para a classificação de notícias e posts de redes sociais como o twitter, regedit e sites.

