cat $1 | grep "Gibbs score: " | awk 'NF>1{print $NF}' > $2
