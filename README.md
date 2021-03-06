# cbic-2021-learning-preferences
Paper published in CBIC2021 entitled "Learning Pairwise Comparisons with Machine Learning for Large-Scale Multi-Criteria Decision Making Problems"
DOI: 10.21528/CBIC2021-13

Abstract: Decision making is a complex task and requires a lot of cognitive effort from the decision maker. Multi-criteria methods, especially those based on pairwise comparisons, such as the Analytic Hierarchic Process (AHP), are not viable for large-scale decision-making problems. For this reason, the aim of this paper is to learn the preferences of the decision-maker using machine learning techniques in order to reduce the number of queries that are necessary in decision problems. We used a recently published parameterized generator of scalable and customizable benchmark problems for many-objective problems as a large-scale data generator. The proposed methodology is an iterative method in which a small subset of solutions are presented to the decision-maker to obtain pairwise judgments. This information is fed to an algorithm that learns the preferences for the remaining pairs in the decision matrix. The Gradient Boosting Regressor was applied in a problem with 5 criteria and 210 solutions. Subsets of 5, 7 and 10 solutions were used in each iteration. The metrics MSE, RMSE, MAPE and R2 were calculated. After the 8th iteration the ranking similarity stabilized, as measured by the tau distance. As the main advantage of the proposed approach is that it was necessary only 8 iterations presenting 5 solutions per time to learn the preferences and get an accurate final ranking. 

@INPROCEEDINGS{CBIC2021-13,
    TITLE=		{Marcos Antonio Alves, Ivan Reinaldo Meneghini and Frederico Gadelha Guimaraes.},
    AUTHOR=		{Learning Pairwise Comparisons with Machine Learning for Large-Scale Multi-Criteria Decision Making Problems},
    PAGES=		{1-7},
    BOOKTITLE=	{Anais do 15 Congresso Brasileiro de Inteligencia Computacional},
    EDITOR=		{Carmelo Jose Albanez Bastos Filho and Hugo Valadares Siqueira and Danton Diego Ferreira and Douglas Wildgrube Bertol and Roberto Celio Limao de Oliveira},
    PUBLISHER=	{SBIC},
    ADDRESS=	{Joinville, SC},
    YEAR=		{2021},
    DOI=    {10.21528/CBIC2021-13}
  }
  
