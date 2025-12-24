dataset 1 ——> batch 1 ——> vae ——> latent\
dataset 2 ——> batch 2 ——> vae ——> latent\
dataset 3 ——> batch 3 ——> vae ——> latent\

$$
x_1,x_2,x_3\xrightarrow{交织+功率归一化} x:[B,num,size] \xrightarrow{信道:awgn+丢包} y \xrightarrow{反交织}
\begin{cases}
\hat{x}_{1窄},\hat{x}_{1中},\hat{x}_{1宽} \xrightarrow{attention解} \hat{x}_1 \xrightarrow{error\ \ detect} \\
\hat{x}_{2窄},\hat{x}_{2中},\hat{x}_{2宽} \xrightarrow{attention解} \hat{x}_2 \xrightarrow{error\ \ detect} \\
\hat{x}_{3窄},\hat{x}_{3中},\hat{x}_{3宽} \xrightarrow{attention解} \hat{x}_3 \xrightarrow{error\ \ detect}
\end{cases}
$$

