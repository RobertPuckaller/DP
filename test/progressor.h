#include <opencv2/opencv.hpp>

class Progressor
{
    int64 start;
    double freq;
    unsigned int total;
    unsigned int count;
    double last;
public:
    Progressor(unsigned int total)
        : start(cv::getTickCount()), freq(cv::getTickFrequency()), total(total), last(-1), count(0)
    {}

    void reportNext(const std::string& txt) {
        count++;
        double sec = (cv::getTickCount() - start) / freq;
        if (sec > last + 0.125) {
            std::cout << "                \r" << std::flush
					  << txt << count << "/" << total << ", "
                      << (count / sec) << "ops, remaining: "
                      << ((total - count) / (count / sec)) << "s";                      
            last = sec;
        }
		if (count == total)  std::cout << "                \r" << std::flush;
    }

    void report(unsigned int i) {
        double sec = (cv::getTickCount() - start) / freq;
        if (sec > last + 0.125) {
            std::cout << i << "/" << total << ", "
                      << (i / sec) << "ops, remaining: "
                      << ((total - i) / (i / sec))
                      << "s                \r"
                      << std::flush;
            last = sec;
        }
    }

	void reset(unsigned int tot) {
		start = cv::getTickCount();
		freq = cv::getTickFrequency();
		total = tot;
		last = -1;
		count = 0;
	}
};
