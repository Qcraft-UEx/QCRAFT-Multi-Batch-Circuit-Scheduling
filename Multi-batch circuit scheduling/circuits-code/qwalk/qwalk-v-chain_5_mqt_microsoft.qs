namespace QcraftExample {
    open Microsoft.Quantum.Intrinsic;
    open Microsoft.Quantum.Canon;
    open Microsoft.Quantum.Measurement;
    open Microsoft.Quantum.Arrays;

    operation qwalk-v-chain_5_mqt_microsoft() : Result[] {
  
        use q = Qubit[5];
        mutable results = new Result[5];


        H(q[3]);
        Controlled X([q[3], q[1]], q[4]);s
        CCNOT(q[2], q[4], q[0]);
        Controlled X([q[3], q[1]], q[4]);
        CCNOT(q[3], q[2], q[1]);
        X(q[1]);
        CNOT(q[3], q[2]);
        X(q[2]);
        X(q[3]);
        Controlled X([q[3], q[1]], q[4]);
        CCNOT(q[2], q[4], q[0]);
        Controlled X([q[3], q[1]], q[4]);
        CCNOT(q[3], q[2], q[1]);
        X(q[1]);
        CNOT(q[3], q[2]);
        X(q[2]);
       
        R(PauliY, PI()/2.0, q[3]);

        Controlled X([q[3], q[1]], q[4]);
        CCNOT(q[2], q[4], q[0]);
        Controlled X([q[3], q[1]], q[4]);
        CCNOT(q[3], q[2], q[1]);
        X(q[1]);
        CNOT(q[3], q[2]);
        X(q[2]);
        X(q[3]);
        Controlled X([q[3], q[1]], q[4]);
        CCNOT(q[2], q[4], q[0]);
        Controlled X([q[3], q[1]], q[4]);
        CCNOT(q[3], q[2], q[1]);
        X(q[1]);
        CNOT(q[3], q[2]);
        X(q[2]);
        R(PauliY, PI()/2.0, q[3]);

        Controlled X([q[3], q[1]], q[4]);
        CCNOT(q[2], q[4], q[0]);
        Controlled X([q[3], q[1]], q[4]);
        CCNOT(q[3], q[2], q[1]);
        X(q[1]);
        CNOT(q[3], q[2]);
        X(q[2]);
        X(q[3]);
        Controlled X([q[3], q[1]], q[4]);
        CCNOT(q[2], q[4], q[0]);
        Controlled X([q[3], q[1]], q[4]);
        CCNOT(q[3], q[2], q[1]);
        X(q[1]);
        CNOT(q[3], q[2]);
        X(q[2]);

        for (i in 0..4) {
            set results w/= i <- M(q[i]);
        }

        ResetAll(q);

        return results;
    }
}
