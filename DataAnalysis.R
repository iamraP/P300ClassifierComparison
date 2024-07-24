library(DescTools)
library(psych)
library("ggpubr")
library('plotly')
library("ggplot2")
library(rstatix)
library(openxlsx)

data_summary <- function(x) {
  m <- mean(x)
  ymin <- m-sd(x)
  ymax <- m+sd(x)
  return(c(y=m,ymin=ymin,ymax=ymax))
}

paired_pairwise_wilcox_test_to_excel <- function(data, output_file) {
  
  test_results <- data %>% pairwise_wilcox_test(Accuracy ~ Classifier, paired=TRUE, p.adjust.method = "bonferroni",alternative="two.sided")
  test_effsize <- data %>% wilcox_effsize(Accuracy ~ Classifier, paired=TRUE, p.adjust.method = "bonferroni",alternative="two.sided")
  
  # Compute mean Accuracy for each classifier
  mean_accuracies <- data %>%
    group_by(Classifier) %>%
    summarise(mean_Accuracy = mean(Accuracy)) %>%
    arrange(desc(mean_Accuracy))
  
  # Order the test results by mean Accuracy
  test_results <- test_results %>%
    mutate(group1 = factor(group1, levels = mean_accuracies$Classifier),
           group2 = factor(group2, levels = mean_accuracies$Classifier)) %>%
    arrange(group1, group2)
  Classifiers <- mean_accuracies$Classifier
  
  # create matrix to export Excel file
  result_matrix <- matrix(NA, nrow = length(Classifiers), ncol = length(Classifiers),
                          dimnames = list(Classifiers, Classifiers))
  result_matrix_W <- matrix(NA, nrow = length(Classifiers), ncol = length(Classifiers),
                          dimnames = list(Classifiers, Classifiers))
  
  result_matrix_r <- matrix(NA, nrow = length(Classifiers), ncol = length(Classifiers),
                            dimnames = list(Classifiers, Classifiers))
  
  
  
  
  
  for (i in seq_len(nrow(test_results))) {
    row <- test_results[i, ]
    row_r <- test_effsize[i, ]
    
    result_matrix[as.character(row$group1), as.character(row$group2)] <- row$p.adj
    result_matrix[as.character(row$group2), as.character(row$group1)] <- row$p.adj
    
    result_matrix_W[as.character(row$group1), as.character(row$group2)] <- row$statistic
    result_matrix_W[as.character(row$group2), as.character(row$group1)] <- row$statistic
    
    result_matrix_r[as.character(row_r$group1), as.character(row_r$group2)] <- row_r$effsize
    result_matrix_r[as.character(row_r$group2), as.character(row_r$group1)] <- row_r$effsize
  }
  
  result_df_r <- as.data.frame(result_matrix_r)
  result_df_r$Accuracy <- mean_accuracies$mean_Accuracy
  result_df_r <- tibble::rownames_to_column(result_df_r, var = "Classifier")
  write.xlsx(result_df_r, paste(output_file,"_r.xls", sep=""), rowNames = TRUE)
  
  # result_df <- as.data.frame(result_matrix)
  # result_df$Accuracy <- mean_accuracies$mean_Accuracy
  # result_df <- tibble::rownames_to_column(result_df, var = "Classifier")
  # write.xlsx(result_df, paste(output_file,"_p.xls", sep=""), rowNames = TRUE)
  # 
  # result_df_W <- as.data.frame(result_matrix_W)
  # result_df_W$Accuracy <- mean_accuracies$mean_Accuracy
  # result_df_W <- tibble::rownames_to_column(result_df_W, var = "Classifier")
  # write.xlsx(result_df_W, paste(output_file,"_W.xls", sep=""), rowNames = TRUE)
  # 
  # return(result_df,result_matrix_W,result_df_r)
}


options(max.print = .Machine$integer.max)

## Basic Findings
amplitudes <- read.csv(r"(..\average_amplitudes_350_600_with_dates.csv)")
amplitudes$Dates <- as.Date(amplitudes$Dates,"%d.%m.%Y")
amplitudes <- subset(amplitudes, Condition=="Target")
summary(amplitudes)
sd(amplitudes$Mean.Amplitude)
mean(amplitudes$Mean.Amplitude)



#drop unecessary columns and Days after Leg Exclusion
online_accuracy <- read.csv(r"(..\online_accuracy.csv)",sep=";")
online_accuracy$Dates <- as.Date(online_accuracy$Datum,"%d.%m.%Y")
#convert string % to actual percentage
online_accuracy$Accuracy <- as.numeric(gsub("%", "", online_accuracy[,"Online.Accuracy"]))/100

sd(online_accuracy$Accuracy)
mean(online_accuracy$Accuracy)

#### Classification ####
classifiers <- read.csv("C:\\Users\\User01\\Documents\\Software\\P300ClassifierComparison\\created_data\\Classifier_Results\\current\\accuracies.csv")

classifiers$Classifier <- as.factor(classifiers$Classifier)

#### H1 ####
#Calibration simmilar to aqcuistion

set_h1 <- subset(classifiers, Ep2Avg==8 & Condition=="sess3")
shapiro.test(set_h1$Accuracy) # not normal distributed
hist(set_h1$Accuracy, breaks=50) # negativly skewed, and pointy?
PlotQQ(set_h1$Accuracy)# seems skewed
describeBy(set_h1$Accuracy, set_h1$Classifier) #  okay i think
LeveneTest(set_h1$Accuracy, set_h1$Classifier) #

#Significance?
kruskal.test(set_h1$Accuracy~set_h1$Classifier)

 paired_pairwise_wilcox_test_to_excel(set_h1, "Sess3")
 
 ggplot(data=set_h1,
       mapping = aes(x=Classifier,
                     y=Accuracy))+
  geom_boxplot(size=1)+
  geom_point(color="steelblue",size=1.5,width = 0.1)+
  theme_classic2()+
  theme(
    text=element_text(color="steelblue", size=20,margin=14)
  )



#### H2 ####
#Calibration single trial calibration

set_h2 <- subset(classifiers, Ep2Avg==8 & Condition=="sess1")
set_h2[is.na(set_h2)] <- 0

shapiro.test(set_h2$Accuracy) # not normal distributed
hist(set_h2$Accuracy, breaks=50) # negativly skewed, and pointy?
PlotQQ(set_h2$Accuracy)# seems skewed
describeBy(set_h2$Accuracy, set_h2$Classifier) #  okay i think
LeveneTest(set_h2$Accuracy, set_h2$Classifier) #

#Significance?
kruskal.test(set_h2$Accuracy~set_h2$Classifier)
paired_pairwise_wilcox_test_to_excel(set_h2, "Sess1")

#### H3 ####
#Calibration transfer condition three session calibration

set_h3 <- subset(classifiers, Ep2Avg==8 & (Condition=="single1_A" | Condition=="single1_B"))

shapiro.test(set_h3$Accuracy) # not normal distributed
hist(set_h3$Accuracy, breaks=50) # negativly skewed, and pointy?
PlotQQ(set_h3$Accuracy)# seems skewed
describeBy(set_h3$Accuracy, set_h3$Classifier) #  okay i think
LeveneTest(set_h3$Accuracy, set_h3$Classifier) #

#Significance?
kruskal.test(set_h3$Accuracy~set_h3$Classifier)
paired_pairwise_wilcox_test_to_excel(set_h3, "Trans1")




#### H4 ####
#Calibration transfer condition single session calibration

set_h4 <- subset(classifiers, Ep2Avg==8 & (Condition=="single3_A"| Condition=="single3_B"))

shapiro.test(set_h4$Accuracy) # not normal distributed
hist(set_h4$Accuracy, breaks=50) # negativly skewed, and pointy?
PlotQQ(set_h4$Accuracy)# seems skewed
describeBy(set_h4$Accuracy, set_h4$Classifier) #  okay i think
LeveneTest(set_h4$Accuracy, set_h4$Classifier) #

#Significance?
kruskal.test(set_h4$Accuracy~set_h4$Classifier)
paired_pairwise_wilcox_test_to_excel(set_h4, "Trans3")



ggplot(data=set_h4,
       mapping = aes(x=Classifier,
                     y=Accuracy))+
  geom_boxplot(size=1)+
  geom_jitter(color="steelblue",size=1.5,width = 0.1)+
  theme_classic2()+
  theme(
    text=element_text(color="steelblue", size=20,margin=14)
  )





