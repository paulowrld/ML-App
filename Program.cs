using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using Accord.Neuro;
using Accord.Neuro.Learning;

class ML
{
    // -------------------------------------------------------------------------
    //  Listas de categorias fixas
    //  A ordem NÃO deve ser alterada: cada posição vira um “bit” no vetor final.
    // -------------------------------------------------------------------------
    static readonly string[] Jobs       = { "admin.", "unknown", "unemployed", "management", "housemaid", "entrepreneur", "student", "blue-collar", "self-employed", "retired", "technician", "services" };
    static readonly string[] Maritals   = { "married", "divorced", "single" };
    static readonly string[] Educations = { "unknown", "secondary", "primary", "tertiary" };
    static readonly string[] Contacts   = { "unknown", "telephone", "cellular" };
    static readonly string[] Months     = { "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec" };
    static readonly string[] POutcomes  = { "unknown", "other", "failure", "success" };

    static void Main()
    {
        // ---------------------------------------------------------------------
        // 1) Carrega conjunto de treinamento e converte registros em vetores
        //    numéricos prontos para a rede (X) + rótulos (Y).
        // --------------------------------------------------------------------- 
        var (trainX, trainY) = LoadDataset("bank.csv");

        var positives = trainX
            .Select((x, i) => (x, y: trainY[i][0]))
            .Where(t => t.y == 1.0)
            .ToList();

        int factor = 5;
        var newX = new List<double[]>(trainX);
        var newY = new List<double[]>(trainY);

        for (int f = 0; f < factor; f++)
            foreach (var (x, y) in positives)
            {
                newX.Add((double[])x.Clone());
                newY.Add(new[] { 1.0 });
            }

        trainX = newX.ToArray();
        trainY = newY.ToArray();

        // ---------------------------------------------------------------------
        // 2) Normaliza features CONTÍNUAS (média 0, desvio 1) — essencial para
        //    acelerar e estabilizar o treinamento da rede.
        //    Guardamos médias e desvios para reaplicar depois.
        // ---------------------------------------------------------------------
        Normalize(trainX, out var means, out var stds);

        // ---------------------------------------------------------------------
        // 3) Carrega conjunto de validação EXTERNA (dados nunca vistos) e aplica
        //    a MESMA normalização — evita vazamento de informação (“data leak”).
        // ---------------------------------------------------------------------
        var (validX, validY) = LoadDataset("bank-full.csv");
        ApplyNormalization(validX, means, stds);

    // -----------------------------------------------------------------------------
    //  Construção da REDE NEURAL (feed-forward) usando Accord.NET
    // -----------------------------------------------------------------------------
    //  ActivationNetwork
    //  -----------------
    //  • Classe que implementa um "Multi-Layer Perceptron" (MLP), isto é, uma
    //    rede neural em camadas totalmente conectadas onde a informação flui
    //    APENAS para frente (não há ciclos).
    //  • O 1º parâmetro é o **tipo de função de ativação** que será usada em
    //    TODOS os neurônios de TODAS as camadas (exceto a de entrada, pois esta
    //    apenas “replica” os valores).
    //  • Os parâmetros seguintes listam o **número de neurônios em cada camada**:
    //
    //        [0]  tamanho da *camada de entrada*   →  = nº de features do dataset
    //        [1]  tamanho da 1ª camada oculta
    //        [2]  tamanho da 2ª camada oculta
    //        [3]  tamanho da camada de saída       →  normalmente 1 ou + classes
    //
    //  SigmoidFunction
    //  ---------------
    //  • F(x) = 1 / (1 + e^(-x))
    //  • Propriedades:
    //       – Saída contínua no intervalo (0,1)  → interpretamos como probabilidade
    //       – Derivada simples (F * (1 − F)), fundamental para Backpropagation
    //       – “Suaviza” valores extremos, evitando explosões no gradiente
    //
    //    É a escolha clássica para **classificação binária** quando a camada de
    //    saída tem 1 neurônio: F>0.5 (ou limiar customizado) → classe positiva.
    //
    //    Obs.: em redes modernas usa-se muito ReLU/ReLU6 nas ocultas, mas Accord
    //    fornece Sigmoid por padrão e, para conjuntos tabulares pequenos, ela
    //    funciona bem.
    //
    //  Arquitetura escolhida:
    //  ----------------------
    //    • trainX[0].Length  → nº de colunas depois do one-hot + numéricas
    //    • 16 neurônios      → 1ª camada oculta: suficiente p/ capturar padrões
    //    •  8 neurônios      → 2ª camada oculta: refina combinações não lineares
    //    •  1 neurônio       → saída: probabilidade de o cliente dizer "yes"
    //
    //  Ajustar esses números é tarefa de “tuning”.  Aqui usamos um modelo leve
    //  (≈ 16*F + 8*16 + 1*8 parâmetros) que treina rápido e costuma generalizar
    //  bem para datasets de poucas dezenas de milhares de linhas.
    // -----------------------------------------------------------------------------
    var net = new ActivationNetwork(
                new SigmoidFunction(),  // função de ativação de TODOS os neurônios
                trainX[0].Length,       // entrada: vetor de features
                16,                     // hidden layer 1
                8,                      // hidden layer 2
                1                       // saída: 1 neurônio → probabilidade
            );

    // -----------------------------------------------------------------------------
    // 1) Nguyen-Widrow weight initialization
    // -----------------------------------------------------------------------------
    new NguyenWidrow(net).Randomize();    
    /*  O QUE É?
        • Método clássico (1989) para “dar o chute inicial” nos pesos de um
            Multi-Layer Perceptron.
        • Trabalha camada a camada: gera valores aleatórios uniformes no
            intervalo (-0.5, 0.5) e depois escala cada neurônio por um fator β:

                    β = 0.7 · (k)^(1/size_in)

            onde k = número de neurônios na camada.  
            Esse fator garante que a **norma** do vetor de pesos de cada neurônio
            seja ≈ 0.7, distribuindo-os sobre a superfície de uma hiperesfera.

        POR QUE USAR?
        • Evita que todas as ativações saiam saturadas (≈0 ou ≈1) logo no
            primeiro forward pass – problema comum quando se usa apenas
            aleatoriedade uniforme ou normal sem escala.
        • Mantém sinais variados (positivos/negativos) e “empurra” a rede para
            uma região do espaço de pesos onde o gradiente não é nem muito fraco
            nem explosivo, acelerando a convergência nas primeiras épocas.
    */


    // -----------------------------------------------------------------------------
    // 2) Resilient Backpropagation (RProp) – algoritmo de treinamento
    // -----------------------------------------------------------------------------
    var teacher = new ResilientBackpropagationLearning(net);
    /*  O QUE É?
        • Variante do backprop standard apresentada por Riedmiller & Braun (1993).
        • Em vez de multiplicar o gradiente por uma *taxa de aprendizado global*,
            RProp mantém um **passo de atualização (Δᵂ)** individual para
            cada peso W_ij e o ajusta apenas com base no SINAL do gradiente:

            se ∂E/∂W muda de sinal ➜ passo diminui   (gradiente passou do mínimo)
            se mantém sinal         ➜ passo aumenta   (estamos indo na direção boa)

            Assim, o tamanho do passo é **desacoplado** da magnitude do gradiente.

        VANTAGENS PRÁTICAS
        • Não exige “tuning” fino de learning rate — poucas redes ficam instáveis.
        • Convergência mais rápida que backprop padrão em problemas tabulares
            pequenos/médios, como o dataset *bank*.
        • Cada peso adapta-se sozinho; regiões rasas ganham passos maiores,
            regiões íngremes ganham passos menores.

        PARÂMETROS DEFAULT (Accord):
        • Δ₀   = 0.1        (passo inicial)
        • η⁺   = 1.2        (fator de aumento)
        • η⁻   = 0.5        (fator de redução)
        • Δmax = 50.0
        • Δmin = 1e-6
        Estes funcionam bem na maioria dos casos; só precisam de ajustes quando
        o erro oscila muito ou a convergência é lenta.
    */


        // ---------------------------------------------------------------------
        // 5) Loop de treinamento (épocas) até erro ficar <0.01 OU atingir 4 000.
        // ---------------------------------------------------------------------
        int epoch = 0;
        double error;
        do
        {
            // RunEpoch treina em TODO o dataset e devolve erro médio da época.
            error = teacher.RunEpoch(trainX, trainY);

            if (epoch % 50 == 0)
                Console.WriteLine($"Época {epoch}, erro: {error:F4}");

            epoch++;
        } while (error > 0.01 && epoch < 4000);

        // ---------------------------------------------------------------------
        // 6) Avaliação final na base “bank-full” (totalmente fora de amostragem)
        // ---------------------------------------------------------------------
        Console.WriteLine("\n== Métricas em bank-full (validação externa) ==");
        PrintMetrics(validX, validY, net);
    }

    // =========================================================================
    //  Lê arquivo CSV, faz parsing de colunas, aplica one-hot nas categóricas e
    //  devolve:
    //     X = double[][]  (amostras × features)
    //     Y = double[][]  (amostras × 1)  [0] = “no”, [1] = “yes”
    // =========================================================================
    static (double[][] X, double[][] Y) LoadDataset(string path)
    {
        var inputs  = new List<double[]>();
        var outputs = new List<double[]>();

        using var reader = new StreamReader(path);
        reader.ReadLine();                         // pula cabeçalho

        while (!reader.EndOfStream)
        {
            // Remove aspas de cada campo e separa por ‘;’
            var split = reader.ReadLine()?
                           .Split(';')
                           .Select(s => s.Replace("\"", ""))
                           .ToArray();

            if (split == null || split.Length == 0) continue;

            // ----------------- Conversão de colunas numéricas puras -----------
            double age      = double.Parse(split[0]);
            double balance  = double.Parse(split[5]);
            double day      = double.Parse(split[9]);
            double campaign = double.Parse(split[12]);
            double pdays    = double.Parse(split[13]);
            double previous = double.Parse(split[14]);

            // ----------------- Booleans → 0/1 ---------------------------------
            double defaultYes = split[4] == "yes" ? 1 : 0;
            double housingYes = split[6] == "yes" ? 1 : 0;
            double loanYes    = split[7] == "yes" ? 1 : 0;

            // Monta vetor de features
            var feat = new List<double>
            {
                age, balance, day,
                campaign, pdays, previous,
                defaultYes, housingYes, loanYes
            };

            // ----------------- One-hot (cada categoria vira vários 0/1) -------
            feat.AddRange(OneHot(split[1],  Jobs));
            feat.AddRange(OneHot(split[2],  Maritals));
            feat.AddRange(OneHot(split[3],  Educations));
            feat.AddRange(OneHot(split[8],  Contacts));
            feat.AddRange(OneHot(split[10].ToLower(), Months));
            feat.AddRange(OneHot(split[15], POutcomes));

            inputs.Add(feat.ToArray());

            // Saída desejada: 1 se “yes”, senão 0
            outputs.Add(new[] { split[16] == "yes" ? 1.0 : 0.0 });
        }
        return (inputs.ToArray(), outputs.ToArray());
    }

    // -----------------------------------------------------------------------------
    //  OneHot
    //  ------
    //  Converte um valor categórico (string) em **vetor binário esparso** do tipo
    //  one-hot.  Exemplo com a lista Months:
    //      value = "mar"     categories = ["jan","feb","mar","apr",...]
    //      retorno → [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    //
    //  Por que usar one-hot?
    //    • Redes neurais trabalham apenas com números.  Strings precisam virar
    //      números para entrar como features.
    //    • Indexar categorias como 0,1,2… seria perigoso: a rede encararia isso
    //      como grandeza ordinal (“abr” > “jan”), o que não faz sentido.
    //    • One-hot remove essa hierarquia implícita; cada categoria ganha um “bit”.
    //
    //  Observação importante:
    //    • A ORDEM em `categories` deve ser fixada no início do projeto e usada
    //      em TODO o pipeline (treino, validação, produção).  Se trocar ordem ou
    //      tamanho, os pesos aprendidos deixam de corresponder às posições corretas.
    //    • Se `value` não estiver em `categories`, devolvemos um vetor TODO zero
    //      (equivale a “categoria desconhecida”).  Isso evita IndexOutOfRange.
    // -----------------------------------------------------------------------------
    static double[] OneHot(string value, string[] categories)
    {
        // Cria vetor cheio de zeros (length = nº categorias)
        var v = new double[categories.Length];

        // Procura posição da string na lista de categorias
        int idx = Array.IndexOf(categories, value);

        // Se encontrou, coloca `1` na posição correspondente
        if (idx >= 0) v[idx] = 1;

        // Se não encontrou, vetor continua com todos zeros
        return v;
    }

    // -----------------------------------------------------------------------------
    //  Normalize
    //  ---------
    //  Objetivo: colocar TODAS as features na mesma escala (média = 0, desvio = 1)
    //            usando o z-score. Isso acelera o treinamento e evita que colunas
    //            numéricas de grande magnitude “domin(em)” o gradiente.
    //
    //        z = (x − μ) / σ
    //
    //  Entradas:
    //    • data  : double[][]  (linhas = amostras, colunas = features)
    //  Saídas (out):
    //    • means : média de cada coluna  (μ₁ … μ_d)           — usada depois
    //    • stds  : desvio-padrão de cada coluna (σ₁ … σ_d)    — idem
    //
    //  Passos:
    //    1) Calcula média de cada coluna
    //    2) Calcula desvio-padrão de cada coluna
    //         • adiciona 1e-9 p/ garantir σ>0 (segurança numérica)
    //    3) Converte o PRÓPRIO conjunto (in-place) para z-scores
    // -----------------------------------------------------------------------------
    static void Normalize(double[][] data, out double[] means, out double[] stds)
    {
        int n = data.Length;        // número de amostras (linhas)
        int d = data[0].Length;     // número de features (colunas)

        means = new double[d];
        stds  = new double[d];

        // 1) MÉDIAS --------------------------------------------------------------
        for (int j = 0; j < d; j++)
        {
            double sum = 0;
            for (int i = 0; i < n; i++)
                sum += data[i][j];

            means[j] = sum / n;
        }

        // 2) DESVIOS-PADRÃO ------------------------------------------------------
        for (int j = 0; j < d; j++)
        {
            double sumSq = 0;
            for (int i = 0; i < n; i++)
            {
                double diff = data[i][j] - means[j];
                sumSq += diff * diff;
            }

            stds[j] = Math.Sqrt(sumSq / (n - 1)) + 1e-9;
        }

        // 3) NORMALIZAÇÃO IN-PLACE ----------------------------------------------
        for (int i = 0; i < n; i++)
            for (int j = 0; j < d; j++)
                data[i][j] = (data[i][j] - means[j]) / stds[j];
    }


    // -----------------------------------------------------------------------------
    //  ApplyNormalization
    //  ------------------
    //  Recebe:
    //    • data  : matriz de amostras que ainda NÃO foi normalizada (cada linha = uma amostra, cada coluna = uma feature)
    //    • means : médias calculadas NO CONJUNTO DE TREINO
    //    • stds  : desvios-padrão calculados NO CONJUNTO DE TREINO
    //
    //  Para cada elemento data[i][j] aplica a transformação z-score:
    //
    //        z = (x − μ) / σ
    //
    //  Por que usar as MESMAS médias/desvios do treino?
    //    → Evita “data leakage”.  Se você recalculasse média e desvio usando
    //      o conjunto de validação/teste, estaria misturando informação que a
    //      rede não deveria conhecer durante o treinamento.
    //
    //  Resultado: conjunto externo passa a estar na mesma escala (média 0, desvio 1)
    //             que os dados que treinaram a rede — condição básica para que
    //             os pesos aprendidos sejam aplicáveis.
    // -----------------------------------------------------------------------------
    static void ApplyNormalization(double[][] data, double[] means, double[] stds)
    {
        // Loop duplo: percorre cada amostra (i) e cada feature (j)
        for (int i = 0; i < data.Length; i++)
            for (int j = 0; j < means.Length; j++)
                // z-score: desloca pela média e divide pelo desvio
                data[i][j] = (data[i][j] - means[j]) / stds[j];
    }


    // -----------------------------------------------------------------------------
    //  Calcula e imprime a Matriz de Confusão + métricas de classificação.
    //  Passos principais:
    //    1) Gera as predições da rede e incrementa a matriz m[exp][pred]
    //    2) Mostra a matriz em formato tabular
    //    3) Extrai TN / FP / FN / TP para calcular
    //         • Acurácia (= acertos / total)
    //         • Precisão  (= TP / (TP+FP))
    //         • Revocação (= TP / (TP+FN))
    //         • F1-Score  (harmônica de Precisão e Revocação)
    //  Observação: usamos limiar de 0.30 em vez de 0.50 para privilegiar recall
    //              — um corte mais baixo produz mais “yes”, reduzindo FNs.
    // -----------------------------------------------------------------------------
    static void PrintMetrics(double[][] X, double[][] Y, ActivationNetwork net)
    {
        // m[expected][predicted]
        // expected (linha)  : 0 = classe “no”,  1 = classe “yes”
        // predicted (coluna): 0 ou 1
        // => m[0][0] = TN,  m[0][1] = FP,  m[1][0] = FN,  m[1][1] = TP
        int[][] m = { new int[2], new int[2] };

        for (int i = 0; i < X.Length; i++)
        {
            int exp = (int)Y[i][0];              // rótulo real       (0 ou 1)
            // A rede devolve um valor ∈ (0,1) (sigmoid).  >0.3 ⇒ classe 1.
            int pred = net.Compute(X[i])[0] > 0.30 ? 1 : 0;
            m[exp][pred]++;                      // acumula contagem
        }

        // ------------------------- Saída da matriz ------------------------------
        Console.WriteLine("Matriz de Confusão:");
        Console.WriteLine($"          Pred 0  Pred 1");
        Console.WriteLine($"Exp 0  {m[0][0],7} {m[0][1],7}");
        Console.WriteLine($"Exp 1  {m[1][0],7} {m[1][1],7}");

        // ------------------------- Métricas derivadas ---------------------------
        // Aliases para clareza
        double TN = m[0][0], FP = m[0][1], FN = m[1][0], TP = m[1][1];
        double total = TN + FP + FN + TP;        // nº amostras

        // Acurácia: fração de predições corretas
        double acc = (TN + TP) / total;

        // Precisão: dentre predições positivas, quantas são corretas?
        double precision = TP / (TP + FP + 1e-9);

        // Revocação (Recall ou Sensibilidade): dentre TODOS os positivos reais,
        // Quantos foram encontrados?  Mais alta ⇒ menos FN.
        double recall = TP / (TP + FN + 1e-9);

        // F1-score: média harmônica de Precisão e Revocação —
        // Punindo caso uma delas seja baixa.
        // 1e-9 para evitar divisão por zero.
        double f1 = 2 * precision * recall / (precision + recall + 1e-9);

        // ------------------------- Impressão final ------------------------------
        Console.WriteLine($"\nAcurácia : {acc:F4}");
        Console.WriteLine($"Precisão : {precision:F4}");
        Console.WriteLine($"Revocação: {recall:F4}");
        Console.WriteLine($"F1 Score : {f1:F4}");
    }
}
