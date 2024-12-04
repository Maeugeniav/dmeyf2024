# --- Code Cell ---
# limpio la memoria
format(Sys.time(), "%a %b %d %X %Y")
rm(list = ls(all.names = TRUE)) # remove all objects
gc(full = TRUE, verbose= FALSE) # garbage collection

dir.create("~/buckets/b1/exp/lineademuerte4/", showWarnings = FALSE)
setwd("~/buckets/b1/exp/lineademuerte4/")

# --- Code Cell ---
require("data.table")

# leo el dataset
dataset <- fread("~/buckets/b1/datasets/competencia_03_crudo.csv.gz")

# calculo el periodo0 consecutivo
setorder(dataset, numero_de_cliente, foto_mes)
dataset[, periodo0 := as.integer(foto_mes / 100) * 12 + foto_mes %% 100]

# calculo topes
periodo_ultimo <- dataset[, max(periodo0)]
periodo_anteultimo <- periodo_ultimo - 1

# calculo los leads de orden 1 y 2
dataset[, c("periodo1", "periodo2") :=
          shift(periodo0, n = 1:2, fill = NA, type = "lead"), numero_de_cliente]

# assign most common class values = "CONTINUA"
dataset[periodo0 < periodo_anteultimo, clase_ternaria := "CONTINUA"]

# calculo BAJA+1
dataset[periodo0 < periodo_ultimo &
          (is.na(periodo1) | periodo0 + 1 < periodo1),
        clase_ternaria := "BAJA+1"]

# calculo BAJA+2
dataset[periodo0 < periodo_anteultimo & (periodo0 + 1 == periodo1) &
          (is.na(periodo2) | periodo0 + 2 < periodo2),
        clase_ternaria := "BAJA+2"]

dataset[, c("periodo0", "periodo1", "periodo2") := NULL]

# --- Code Cell ---
tbl <- dataset[, .N, list(foto_mes, clase_ternaria)]
setorder(tbl, foto_mes, clase_ternaria)
tbl

# --- Code Cell ---
# Feature Engineering Historico
cols_lagueables <- copy(setdiff(
  colnames(dataset),
  c("numero_de_cliente", "foto_mes", "clase_ternaria")
))

dataset[, paste0(cols_lagueables, "_lag1") := shift(.SD, 1, NA, "lag"),
        by = numero_de_cliente,
        .SDcols = cols_lagueables
]

dataset[, paste0(cols_lagueables, "_lag2") := shift(.SD, 2, NA, "lag"),
        by = numero_de_cliente,
        .SDcols = cols_lagueables
]

# agrego los delta lags de orden 1
for (vcol in cols_lagueables) {
  dataset[, paste0(vcol, "_delta1") := get(vcol) - get(paste0(vcol, "_lag1"))]
  dataset[, paste0(vcol, "_delta2") := get(vcol) - get(paste0(vcol, "_lag2"))]
}

# --- Code Cell ---
ncol(dataset)
colnames(dataset)

# --- Code Cell ---
GLOBAL_semilla <- 111119

# --- Code Cell ---
campos_buenos <- copy(setdiff(
  colnames(dataset), c("clase_ternaria"))
)

set.seed(GLOBAL_semilla, kind = "L'Ecuyer-CMRG")
dataset[, azar := runif(nrow(dataset))]

dfuture <- dataset[foto_mes == 202109]

# undersampling de los CONTINUA al 8%
dataset[, fold_train := foto_mes <= 202107 &
          !(foto_mes %in% c(202006, 202004, 202003, 202005, 202104, 201910, 201905))]

dataset[, clase01 := ifelse(clase_ternaria == "CONTINUA", 0, 1)]

require("lightgbm")

# dejo los datos en el formato que necesita LightGBM
dvalidate <- lgb.Dataset(
  data = data.matrix(dataset[foto_mes == 202107, campos_buenos, with = FALSE]),
  label = dataset[foto_mes == 202107, clase01],
  free_raw_data = TRUE
)

# aqui se hace la magia informatica con los pesos para poder reutilizar
#  el mismo dataset para training y final_train
dtrain <- lgb.Dataset(
  data = data.matrix(dataset[fold_train == TRUE, campos_buenos, with = FALSE]),
  label = dataset[fold_train == TRUE, clase01],
  weight = dataset[fold_train == TRUE, ifelse(foto_mes <= 202106, 1.0, 0.0)],
  free_raw_data = TRUE
)

rm(dataset)
gc(full = TRUE, verbose = FALSE) # garbage collection

# --- Code Cell ---
nrow(dfuture)
nrow(dvalidate)
nrow(dtrain)

# --- Code Cell ---
# parametros basicos del LightGBM
param_basicos <- list(
  objective = "binary",
  metric = "auc",
  first_metric_only = TRUE,
  boost_from_average = TRUE,
  feature_pre_filter = FALSE,
  verbosity = -100,
  force_row_wise = TRUE, # para evitar warning
  seed = GLOBAL_semilla,
  max_bin = 31,
  learning_rate = 0.03,
  feature_fraction = 0.5
)

EstimarGanancia_AUC_lightgbm <- function(x) {
  message(format(Sys.time(), "%a %b %d %X %Y"))
  param_train <- list(
    num_iterations = 2048,
    early_stopping_rounds = 200
  )
  
  param_completo <- c(param_basicos, param_train, x)
  
  modelo_train <- lgb.train(
    data = dtrain,
    valids = list(valid = dvalidate),
    eval = "auc", 
    param = param_completo,
    verbose = -100
  )
  
  AUC <- modelo_train$record_evals$valid$auc$eval[[modelo_train$best_iter]]
  attr(AUC, "extras") <- list("num_iterations" = modelo_train$best_iter)
  
  rm(modelo_train)
  gc(full = TRUE, verbose = FALSE)
  
  return(AUC)
}

# --- Code Cell ---
# paquetes necesarios para la Bayesian Optimization
require("DiceKriging")
require("mlrMBO")

configureMlr(show.learner.output = FALSE)

# configuro la busqueda bayesiana
obj.fun <- makeSingleObjectiveFunction(
  fn = EstimarGanancia_AUC_lightgbm,
  minimize = FALSE,
  noisy = FALSE,
  par.set = makeParamSet(
    makeIntegerParam("num_leaves", lower = 8L, upper = 1024L),
    makeIntegerParam("min_data_in_leaf", lower = 64L, upper = 8192L)
  ),
  has.simple.signature = FALSE
)

ctrl <- makeMBOControl(
  save.on.disk.at.time = 600,
  save.file.path = "lineademuerte.RDATA"
)

ctrl <- setMBOControlTermination(ctrl, iters = 10)
ctrl <- setMBOControlInfill(ctrl, crit = makeMBOInfillCritEI())

surr.km <- makeLearner(
  "regr.km",
  predict.type = "se",
  covtype = "matern3_2",
  control = list(trace = TRUE)
)

bayesiana_salida <- mbo(obj.fun, learner = surr.km, control = ctrl)

tb_bayesiana <- as.data.table(bayesiana_salida$opt.path)
setorder(tb_bayesiana, -y, -num_iterations)
mejores_hiperparametros <- tb_bayesiana[1, list(num_leaves, min_data_in_leaf, num_iterations)]

set_field(dtrain, "weight", rep(1.0, nrow(dtrain)))

param_final <- c(param_basicos, mejores_hiperparametros)

final_model <- lgb.train(
  data = dtrain,
  param = param_final,
  verbose = -100
)

prediccion <- predict(
  final_model,
  data.matrix(dfuture[, campos_buenos, with = FALSE])
)

tb_entrega <- dfuture[, list(numero_de_cliente)]
tb_entrega[, prob := prediccion]
setorder(tb_entrega, -prob)
tb_entrega[, prob := NULL]
tb_entrega[, Predicted := 0L]
tb_entrega[1:11000, Predicted := 1L]

fwrite(tb_entrega, file = "lineademuerte_110004.csv")

# --- Code Cell ---
# Generar el gráfico de ganancias vs cortes
cortes <- seq(9000, 12000, by = 500)
ganancias <- numeric(length(cortes))

for (i in seq_along(cortes)) {
  tb_entrega[, Predicted := 0L]
  tb_entrega[1:cortes[i], Predicted := 1L]
  ganancias[i] <- sum(tb_entrega[Predicted == 1L, Predicted]) # Ajusta con la métrica real
}

# Especificar el directorio de salida para el PDF
output_dir <- dirname("lineademuerte_11000.csv")
output_pdf <- file.path(output_dir, "ganancia_vs_corte.pdf")

# Generar y guardar el gráfico
pdf(output_pdf)
plot(
  cortes, ganancias,
  type = "b",  # Línea y puntos
  col = "blue",
  pch = 19,
  xlab = "Corte (número de clientes)",
  ylab = "Ganancia estimada",
  main = "Ganancia vs Corte"
)
grid()
dev.off()  # Cerrar el archivo PDF
