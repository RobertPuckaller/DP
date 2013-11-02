#include <opencv2/opencv.hpp>
#include <string>
#include <fstream>

using namespace cv;

/**
 * A timer class which can be used to report how much a task has taken.
 * On destruction it will report the total time this object was alive.
 */
class Timer {
    int64 startTime, lastTime;
    std::string name;
    std::ostream &out;
    bool silent;
public:
	Timer(std::string name, std::ostream& out) : name(name), out(out), silent(false) {
        startTime = lastTime = getTickCount();
    }

    Timer(std::string name, bool canOutput = true) : name(name), out(std::cout), silent(!canOutput) {
        startTime = lastTime = getTickCount();
    }

    ~Timer() {
        report("total", true);
    }

    /**
     * Report the time in ms from the last report or from the beginning in form
     * "[<name>/<part>] took <time>ms" to the standard error output.
     * @param part The text to display.
     * @param fromStart Whether to count from last event (false) or from the time
     *                  this object was created. Defaults to false.
     */
    double report(const std::string& part, bool fromStart = false) {
        int64 start = fromStart ? startTime : lastTime;
        int diff = (getTickCount() - start) / getTickFrequency() * 1000;
        if (!silent) {
            out << "[" << name << "/" << part << "] took ";
			convertunits(diff);
		}
        lastTime = getTickCount();
        return diff;
    }

	void convertunits(int diff) {
		if (diff > 3600000) {
			out << diff / 3600000 << "h ";
			diff = diff % 3600000;
		}
		if (diff > 60000) {
			out << diff / 60000 << "m ";
			diff = diff % 60000;
		}
		if (diff > 1000) {
			out << diff / 1000 << "s ";
			diff = diff % 1000;
		}
		out << diff << "ms" << std::endl;
	}
};
