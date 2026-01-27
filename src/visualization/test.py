from record import RecordBuffer
from plot import EasyLinePlot


if __name__ == "__main__":
    buf = RecordBuffer()

    # random record 100 entries
    for i in range(100):
        if i < 50:
            buf.append(step=i)
        buf.append(value1=i * 0.5 + (i % 5))
        buf.append(value2=i * 0.3 + (i % 3) * 2)

    # create plot from buffer data
    plot = EasyLinePlot(buf.data)
    plot.plot(title="Random Data", xlabel="Step", ylabel="Value")
