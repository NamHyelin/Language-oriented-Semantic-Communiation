import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

qam = 16

ebno = np.logspace(0.5, 1.0, 5)  # 1.2, 12
snr = 10 * np.log10(ebno)

K = np.sqrt(qam)  # 4
M = 2 ** K
Pm = (1 - 1 / np.sqrt(M)) * erfc(np.sqrt(3 / 2 / (M - 1) * K * ebno))
Ps_qam = 1 - (1 - Pm) ** 2
Pb_qam = Ps_qam / K


if __name__ == "__main__":
    plt.subplot(1,2,1)
    plt.semilogy(snr, Pb_qam, ':b+')
    plt.legend()
    plt.axis([min(snr), max(snr), 1e-6, 1])
    plt.title('16-QAM Modulation (AWGN Channel)')
    plt.xlabel('SNR [dB]')
    plt.ylabel('Bit Error Probability P_b')

    plt.subplot(1,2,2)
    plt.semilogy(snr, Ps_qam, ':b+')
    plt.legend()
    plt.axis([min(snr), max(snr), 1e-6, 1])
    plt.title('16-QAM Modulation (AWGN Channel)')
    plt.xlabel('SNR [dB]')
    plt.ylabel('Symbol Error Probability P_s')

    plt.show()

    print(snr)
    print(Pb_qam)