require(readr)

load_naming = function(){
    msyl = read_csv("../data/sense_graph/sense_graph_monosyll.csv")
    phonetics = read_csv("../data/bopomo-phonetics.csv")
    ambig = read_csv("../data/naming_semanticAmbiguity.csv")
    naming = read_csv("../data/naming_sense_indices.csv")
    print(dim(naming))
    naming$initphone = substr(naming$zhuyin, 1, 1)
    naming = merge(naming, phonetics, by.x="initphone", by.y="X1")
    naming = naming[,-1]
    naming$log_freq = log(naming$Frequency)
    naming$RTinv = -1000/naming$RT
    naming$sf_rmax = log((naming$sfreq_max+1) / (naming$sfreq_sum+1))
    naming = merge(naming, msyl, by.x="Character", by.y="mw")
    naming$rEV = log(naming$mw_nE+1)-log(naming$mw_nV+1)
    naming = merge(naming ,ambig, by.x="Character", by.y="Character")
    names(naming)[names(naming) == "Semantic Ambiguity Rating"] = "sar"
    names(naming)[names(naming) == "Log CD"] = "log_CD"
    return(naming)
}

load_word_ldt = function(){
    bsyl = read_csv("../data/sense_graph/sense_graph_bisyll.csv")
    msyl = read_csv("../data/sense_graph/sense_graph_monosyll.csv")
    clp = read_csv("../data/CLP_sense_indices_2char.csv")
    clp$C1 = substr(clp$Word_Trad, 1, 1)
    clp$C2 = substr(clp$Word_Trad, 2, 2)
    clp$log_freq_W = log(clp$`SS&M-W`)
    clp$log_freq_C1 = log(clp$`SS&M-C1`)
    clp$log_freq_C2 = log(clp$`SS&M-C2`)
    clp$RTinv = -1000/clp$RT
    clp$sf_rmax_W = log((clp$sfreq_max_W+1) / (clp$sfreq_sum_W+1))
    clp$sf_rmax_C1 = log((clp$sfreq_max_C1+1) / (clp$sfreq_sum_C1+1))
    clp$sf_rmax_C2 = log((clp$sfreq_max_C2+1) / (clp$sfreq_sum_C2+1))
    clp = merge(clp, msyl, by.x="C1", by.y="mw")
    clp = merge(clp, msyl, by.x="C2", by.y="mw", suffixes=c("_C1", "_C2"))
    c2_mw_idx = grep("^mw_", names(clp))
    clp = merge(clp, bsyl, by.x="Word_Trad", by.y="word")
    clp$rVE_C1 = log(clp$mw_nV_C1)-log(clp$mw_nE_C1)
    clp$rVE_C2 = log(clp$mw_nV_C2)-log(clp$mw_nE_C2)

    return(clp)
}

load_nonword_ldt = function() {
    nwdata = read_csv("../data/sense_graph/sense_graph_nonword.csv")
    nwdata$RTinv = -1000/nwdata$RT
    nwdata$log_freq_c1 = log(nwdata$`Google-freq-C1`)
    nwdata$log_freq_c2 = log(nwdata$`Google-freq-C2`)
    nwdata$log_c1_sfreq = log(nwdata$c1_sense_freq)
    nwdata$log_c2_sfreq = log(nwdata$c2_sense_freq)
    colnames(nwdata)[colnames(nwdata)=='Stroke-1'] = "stroke_c1"
    colnames(nwdata)[colnames(nwdata)=='Stroke-2'] = "stroke_c2"
    nwdata$min_dist_weighted = with(nwdata, sqrt(log_c1_sfreq+log_c2_sfreq)*min_dist)

    return(nwdata)
}

naming = load_naming()
wdata = load_word_ldt()
nwdata = load_nonword_ldt()
cat("Dataset is loaded to naming, wdata, nwdata")
