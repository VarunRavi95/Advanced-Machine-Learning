num_states <- 10
hidden_states <- paste0('S', 1:num_states)
obs_states <- paste0('s', 1:num_states)

init_prob <- runif(10, min = 0, max = 1)

# Defining the transition matrix dimension: hidden_states x hidden_states

transMat <- matrix(0, nrow = num_states, ncol = num_states)
rownames(transMat) <- hidden_states
colnames(transMat) <- hidden_states

for (i in 1:num_states) {
  transMat[i, i] <- 0.5  # Probability of staying in the current sector
  if (i < num_states) {
    transMat[i, i + 1] <- 0.5  # Probability of moving to the next sector
  }else{
    transMat[i, 1] <- 0.5
  }
}

# Defining emission matrix with dimension: hidden_states x obs_symbols
emissMat <- matrix(c(0.2,0.2,0.2,0.2,0.2,0,0,0,0,0,
                     0,0.2,0.2,0.2,0.2,0.2,0,0,0,0,
                     0,0,0.2,0.2,0.2,0.2,0.2,0,0,0,
                     0,0,0,0.2,0.2,0.2,0.2,0.2,0,0,
                     0,0,0,0,0.2,0.2,0.2,0.2,0.2,0,
                     0,0,0,0,0,0.2,0.2,0.2,0.2,0.2,
                     0.2,0,0,0,0,0,0.2,0.2,0.2,0.2,
                     0.2,0.2,0,0,0,0,0,0.2,0.2,0.2,
                     0.2,0.2,0.2,0,0,0,0,0,0.2,0.2,
                     0.2,0.2,0.2,0,0,0,0,0,0.2,0.2),nrow = 10,byrow = TRUE)
rownames(emissMat) <- obs_states
colnames(emissMat) <- obs_states


hmm_model <- initHMM(States=hidden_states, 
                     Symbols=obs_states, 
                     startProbs=init_prob, 
                     transProbs=transMat, emissionProbs=emissMat)

simulate_HMM <- function(model, time_steps, alt_method = FALSE){
  sim_timeStep <- simHMM(model, length = time_steps)
  sim_actual <- sim_timeStep$states
  sim_obs <- sim_timeStep$observation
  
  Alpha <- exp(forward(hmm = model, observation = sim_obs))
  Beta <- exp(backward(hmm = model, observation = sim_obs))
  
  filtered <- apply(Alpha,2,prop.table)
  filter_pred <- apply(filtered, 2, which.max)
  filter_pred <- sapply(filter_pred, function(x) paste0('s',x))
  
  if (alt_method == FALSE) {
    smoothed <- Alpha*Beta
    smoothed <- apply(smoothed, 2, prop.table)
    smoothed_pred <- apply(smoothed, 2, which.max)
    smoothed_pred <- sapply(smoothed_pred, function(x) paste0('s',x))
  }else{
    # Alternative method to calculate smoothed distribution. Correct?
    smoothed <- posterior(hmm = hmm_model, observation = sim_obs)
    smoothed_pred <- apply(smoothed, 2, which.max)
    smoothed_pred <- sapply(smoothed_pred, function(x) paste0('s',x))
  }
  
  
  viterbi_path <- viterbi(hmm = model, observation = sim_obs)
  
  return(list('actual' = sim_actual, 'obs' = sim_obs,'filter_pred'= filter_pred, 
              'smoothed_pred' = smoothed_pred,
              'viterbi' = viterbi_path, 'filter_mat' = filtered))
}

accuracy <- function(actual, pred){
  accuracy <- sum(diag(table(actual, pred)))/sum(table(actual, pred))
  return(accuracy)
}

sim100_results <- simulate_HMM(model = hmm_model, time_steps = 100, alt_method = TRUE) 


filter_accuracy <- accuracy(actual = sim100_results$actual, pred = sim100_results$filter_pred)
smoothed_accuracy <- accuracy(actual = sim100_results$actual, pred = sim100_results$smoothed_pred)
viterbi_accuracy <- accuracy(actual = sim100_results$actual, pred = sim100_results$viterbi)

# Q5

sim100_samplesRes <- list(
  filter_accs = NULL,
  smooth_accs = NULL,
  viterbi_accs = NULL
)

for (i in 1:10) {
  simulate_hmm_vals <- simulate_HMM(model = hmm_model, time_steps = 100, alt_method = FALSE)
  sim100_samplesRes$filter_accs[i] <- accuracy(actual = simulate_hmm_vals$actual, 
                                               pred = simulate_hmm_vals$filter_pred)
  
  sim100_samplesRes$smooth_accs[i] <- accuracy(actual = simulate_hmm_vals$actual, 
                                               pred = simulate_hmm_vals$smoothed_pred)
  
  sim100_samplesRes$viterbi_accs[i] <- accuracy(actual = simulate_hmm_vals$actual, 
                                               pred = simulate_hmm_vals$viterbi)
}


cat('The mean accuracy of filtered distribution is', mean(sim100_samplesRes$filter_accs), '\n')
cat('The mean accuracy of smoothed distribution is', mean(sim100_samplesRes$smooth_accs), '\n')
cat('The mean accuracy of most probable paths distribution is', mean(sim100_samplesRes$viterbi_accs), '\n')


plot(sim100_samplesRes$filter_accs, type = 'l', col = 'red', ylim = c(0.3,0.85))
lines(sim100_samplesRes$smooth_accs, type = 'l', col = 'blue')
lines(sim100_samplesRes$viterbi_accs, type = 'l', col = 'green')


# Q6


entropies <- c()
new_simHMM <- simHMM(hmm = hmm_model, length = 300)
new_obs <- new_simHMM$observation
Alpha_new <- exp(forward(hmm = hmm_model, observation = new_obs))
filtered <- apply(Alpha_new,2,prop.table)
for (i in seq(30, 300, 5)) {
  
  entropies <- c(entropies, entropy.empirical(filtered[,i]))
}

plot(seq(30, 300, 5),entropies, type = 'l', xlab = 'Time Steps', ylab = 'Entropy')

View(sim100_results$filter_mat)
