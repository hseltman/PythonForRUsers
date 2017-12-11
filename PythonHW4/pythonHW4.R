# Simulation for optional python HW 4
# CMU MSP 36601. H. Seltman, Fall 2017

hw4=read.csv("hw4.csv")
summary(lm(score~I(age-40)+I((age-40)^2)+tx+female, hw4))
with(hw4, plot(score~age, col=as.numeric(tx), pch=16+female, ylim=c(0, 120)))
legend("bottomleft", paste(rep(levels(hw4$tx), each=2), rep(c("M","F"), 3)),
       col=rep(1:3, each=2), pch=rep(16:17, 3))
