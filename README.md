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

> Em seguida realizou-se a separacao da base em features e alvos ao todo tem-se um total de 2225 noticias catalogadas na base.

      print(len(X)):
      2225
  
__Step 03:__ Adiante realizou-se o download das stop words, defini-se stop words como:

> Stop words (ou palavras de parada – tradução livre) são palavras que podem ser consideradas irrelevantes para o conjunto de resultados a ser exibido em uma busca realizada em uma search engine. Exemplos: as, e, os, de, para, com, sem, foi.

mais definicoes em:

https://pythonalgos.com/nlp-stop-words-when-and-why-to-use-them/

> para isso utilizou-se o framework NLTK, mais informacoes disponiveis em:

https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/index.xml

> como a base de noticias esta em ingles utilizaram-se as stopwords em ingles com o comando:

> my_stop_words = set(stopwords.words('english'))

__Step 04:__ Adiante realizou-se o download das stop words, defini-se stop words como:



