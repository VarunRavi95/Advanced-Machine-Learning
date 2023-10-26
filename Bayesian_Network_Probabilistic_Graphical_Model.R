# Question 1
# Running hill-climbing for structure based learning, a score based approach
library(bnlearn)
library(gRain)
set.seed(123)
# Defining empty dag for initial structure
random_graph <- random.graph(nodes = colnames(asia))
plot(random_graph)

dag = hc(asia, start = random_graph, restart = 100, max.iter = 100)
dag
plot(dag)
# graphviz.plot(dag,)

dag_true = model2network("[A][S][T|A][L|S][B|S][D|B:E][E|T:L][X|E]")

plot(dag_true)
par(mfrow = c(1,1))
cp_dag <- cpdag(dag)

#Returns a Matrix/Array of from-to directions of dag
arcs(dag) 
arcs(dag_true)
# Returns a Matrix/Array of colliders/v-structures in the dag
vstructs(dag, arcs = TRUE)

all.equal(dag_true, dag)

# Question 2
train_data <- asia[sample(dim(asia)[1], round(0.8*dim(asia)[1])),]
test_data <- asia[-as.numeric(row.names(train_data)),]

learn_dag <- hc(train_data)
plot(learn_dag)
class(learn_dag)
fitted_dag <- bn.fit(x = learn_dag, data = train_data)
fitted_dag_true <- bn.fit(x = dag_true, data = train_data)

bn.fit.barchart(fitted_dag$S)
bn.fit.dotplot(fitted_dag$S)

# Question 2 & 3

bayesNet_infer <- function(bn_fit, test_data, target_node, 
                           markov_blanket = FALSE,
                           exact_inference = TRUE){
  # bn_fit = structure and parameter learned BN
  # Compiling the fitted Bayesian Network as a grain object
  bn_compile <- compile(as.grain(bn_fit))
  
  if (markov_blanket == FALSE) {
    # Evidence nodes are selected without markov blanket
    obs_node <- setdiff(colnames(test_data), target_node)
    obs_data <- as.data.frame(test_data[,obs_node])
  }else{
    # Evidence Nodes are selected based on markov-blanket of target node
    obs_node <- mb(x = fitted_dag, node = target_node)
    obs_data <- as.data.frame(test_data[,obs_node])
  }
  
  predicted_S <- vector("character", length = nrow(obs_data))
  
  # We perform exact inference using setEvidence(), querygrain() here
  if (exact_inference == TRUE) {
    for (i in 1:nrow(obs_data)) {
      # Set evidence data of evidence nodes.
      set_obs_data <- setEvidence(object = bn_compile, 
                                  nodes = obs_node, states = t(obs_data[i,]))
      #Querying the network for 
      query_result <- querygrain(object = set_obs_data, 
                                 nodes = target_node, type = 'marginal')
      
      predicted_S[i] <- ifelse(query_result$S['no'] > query_result$S['yes'], 
                               'no', 'yes')
    }
    
  }else{
    #We perform approximate inference using cpquery() here.
    for (j in 1:nrow(obs_data)) {
      
      # Perform the conditional probability query of the event given evidence
      cp_query <- cpquery(fitted = fitted_dag, event = (S == 'yes'), 
                          evidence = as.list(obs_data[j,]), method = "lw")
      # cp_dist <- cpdist(fitted_dag, nodes = 'S', evidence = as.list(obs_data[j, ]))
      # Store the result in the results vector
      # cp_results[j] <- mean(cp_dist['S'] == 'yes')
      predicted_S[j] <- ifelse(cp_query > 0.5, 'yes', 'no')
    }
  }
  
  return(predicted_S)
    
}

pred_exactInf <- bayesNet_infer(bn_fit = fitted_dag, test_data = test_data, 
                                target_node = 'S',markov_blanket = FALSE, 
                                exact_inference = TRUE)

pred_approxInf <- bayesNet_infer(bn_fit = fitted_dag, test_data = test_data, 
                                 target_node = 'S',markov_blanket = FALSE, 
                                 exact_inference = FALSE)

pred_markovBlanket_exactInf <- bayesNet_infer(bn_fit = fitted_dag, 
                                              test_data = test_data, 
                                              target_node = 'S',
                                              markov_blanket = TRUE, 
                                              exact_inference = TRUE)

pred_markovBlanket_apprInf <- bayesNet_infer(bn_fit = fitted_dag, 
                                             test_data = test_data, 
                                             target_node = 'S',
                                             markov_blanket = TRUE, 
                                             exact_inference = FALSE)

confusion_matrix <- function(pred, true){
  conf_matrix <- table(true$S, pred)
  accuracy <- (sum(diag(conf_matrix))/sum(conf_matrix))*100
  return(list('Confusion_Matrix' = conf_matrix, 'Accuracy' = accuracy))
}

confusion_matrix(pred = pred_markovBlanket_exactInf, true = test_data)

confusion_matrix(pred = pred_markovBlanket_apprInf, true = test_data)

# Question 4


naiveBayes_classifier <- function(train_data, test_data, model_string, 
                                  target_node){
  naiveBayes_dag <- empty.graph(nodes = colnames(asia))
  modelstring(naiveBayes_dag) <- model_string
  
  naiveBayes_fit <- bn.fit(x = naiveBayes_dag, 
                           data = train_data, 
                           method = 'bayes')
  
  naiveBayes_compile <- compile(object = as.grain(naiveBayes_fit))
  
  obs_node <- setdiff(colnames(test_data), target_node)
  obs_data <- as.data.frame(test_data[,obs_node])
  
  pred_naiveBayes <- vector("character", length = nrow(obs_data))
  
  for (i in 1:nrow(obs_data)) {
    set_obs_data <- setEvidence(object = naiveBayes_compile, nodes = obs_node, 
                                states = t(obs_data[i,]))
    query_result <- querygrain(object = set_obs_data, 
                               nodes = 'S', type = 'marginal')
    pred_naiveBayes[i] <- ifelse(query_result$S['yes'] > 0.5, 
                                 'yes', 'no')
  } 
  return(list(pred_naiveBayes, naiveBayes_dag))
}

naiveBayes_result <- naiveBayes_classifier(train_data = train_data,
                                           model_string = '[S][A|S][T|S][L|S][B|S][E|S][X|S][D|S]',
                                           test_data = test_data, target_node = 'S')

graphviz.plot(naiveBayes_result[[2]])

naiveBaye_cfmat <- confusion_matrix(pred = naiveBayes_result[[1]], true = test_data)

true_naiveBayes <- naive.bayes(x = train_data, training = 'S', 
                               explanatory = setdiff(colnames(test_data), 'S'))
true_pred <- predict(true_naiveBayes, test_data)

true_naiveBayes_cfmat <- confusion_matrix(true_pred, test_data)

# Question1: Is the approximate inference methodology correct?
# Question2: Do both exact and approximate inference give same result
# the unobserved data conditions on the same set of observed nodes?
