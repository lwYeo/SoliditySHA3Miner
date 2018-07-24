#include "solver.h"

namespace OpenCLSolver
{
	Solver::Solver(System::String ^maxDifficulty, System::String ^solutionTemplate, System::String ^kingAddress) :
		ManagedObject(new openCLSolver(ToNativeString(maxDifficulty), ToNativeString(solutionTemplate), ToNativeString(kingAddress)))
	{
		m_managedOnGetWorkPosition = gcnew OnGetWorkPositionDelegate(this, &Solver::OnGetWorkPosition);
		System::IntPtr getWorkPositionStubPtr = System::Runtime::InteropServices::Marshal::GetFunctionPointerForDelegate(m_managedOnGetWorkPosition);
		openCLSolver::GetWorkPositionCallback getWorkPositionFnPtr = static_cast<openCLSolver::GetWorkPositionCallback>(getWorkPositionStubPtr.ToPointer());
		m_Instance->setGetWorkPositionCallback(getWorkPositionFnPtr);
		System::GC::KeepAlive(m_managedOnGetWorkPosition);

		m_managedOnResetWorkPosition = gcnew OnResetWorkPositionDelegate(this, &Solver::OnResetWorkPosition);
		System::IntPtr resetWorkPositionStubPtr = System::Runtime::InteropServices::Marshal::GetFunctionPointerForDelegate(m_managedOnResetWorkPosition);
		openCLSolver::ResetWorkPositionCallback resetWorkPositionFnPtr = static_cast<openCLSolver::ResetWorkPositionCallback>(resetWorkPositionStubPtr.ToPointer());
		m_Instance->setResetWorkPositionCallback(resetWorkPositionFnPtr);
		System::GC::KeepAlive(m_managedOnResetWorkPosition);

		m_managedOnIncrementWorkPosition = gcnew OnIncrementWorkPositionDelegate(this, &Solver::OnIncrementWorkPosition);
		System::IntPtr incrementWorkPositionStubPtr = System::Runtime::InteropServices::Marshal::GetFunctionPointerForDelegate(m_managedOnIncrementWorkPosition);
		openCLSolver::IncrementWorkPositionCallback incrementWorkPositionFnPtr = static_cast<openCLSolver::IncrementWorkPositionCallback>(incrementWorkPositionStubPtr.ToPointer());
		m_Instance->setIncrementWorkPositionCallback(incrementWorkPositionFnPtr);
		System::GC::KeepAlive(m_managedOnIncrementWorkPosition);

		m_managedOnMessage = gcnew OnMessageDelegate(this, &Solver::OnMessage);
		System::IntPtr messageStubPtr = System::Runtime::InteropServices::Marshal::GetFunctionPointerForDelegate(m_managedOnMessage);
		openCLSolver::MessageCallback messageFnPtr = static_cast<openCLSolver::MessageCallback>(messageStubPtr.ToPointer());
		m_Instance->setMessageCallback(messageFnPtr);
		System::GC::KeepAlive(m_managedOnMessage);

		m_managedOnSolution = gcnew OnSolutionDelegate(this, &Solver::OnSolution);
		System::IntPtr solutionStubPtr = System::Runtime::InteropServices::Marshal::GetFunctionPointerForDelegate(m_managedOnSolution);
		openCLSolver::SolutionCallback solutionFnPtr = static_cast<openCLSolver::SolutionCallback>(solutionStubPtr.ToPointer());
		m_Instance->setSolutionCallback(solutionFnPtr);
		System::GC::KeepAlive(m_managedOnSolution);
	}

	Solver::~Solver()
	{
		try { m_Instance->~openCLSolver(); }
		catch(...) {}
	}

	void Solver::preInitialize(bool allowIntel, System::String ^%errorMessage)
	{
		std::string errMsg;
		openCLSolver::preInitialize(allowIntel, errMsg);

		errorMessage = ToManagedString(errMsg);
	}

	System::String ^Solver::getPlatformNames()
	{
		std::string platformName;
		platformName = openCLSolver::getPlatformNames();

		return ToManagedString(platformName);
	}

	int Solver::getDeviceCount(System::String ^platformName, System::String ^%errorMessage)
	{
		std::string errMsg;
		int devCount = openCLSolver::getDeviceCount(ToNativeString(platformName), errMsg);

		errorMessage = ToManagedString(errMsg);
		return devCount;
	}

	System::String ^Solver::getDeviceName(System::String ^platformName, int deviceEnum, System::String ^%errorMessage)
	{
		std::string errMsg, devName;
		devName = openCLSolver::getDeviceName(ToNativeString(platformName), deviceEnum, errMsg);

		errorMessage = ToManagedString(errMsg);
		return ToManagedString(devName);
	}

	void Solver::setCustomDifficulty(uint32_t customDifficulty)
	{
		m_Instance->setCustomDifficulty(customDifficulty);
	}

	void Solver::setSubmitStale(bool submitStale)
	{
		m_Instance->isSubmitStale = submitStale;
	}

	bool Solver::assignDevice(System::String ^platformName, int const deviceID, float const intensity)
	{
		return m_Instance->assignDevice(ToNativeString(platformName), deviceID, intensity);
	}

	bool Solver::isAssigned()
	{
		if (m_Instance == nullptr) return false;
		return m_Instance->isAssigned();
	}

	bool Solver::isAnyInitialised()
	{
		if (m_Instance == nullptr) return false;
		return m_Instance->isAnyInitialised();
	}

	bool Solver::isMining()
	{
		if (m_Instance == nullptr) return false;
		return m_Instance->isMining();
	}

	bool Solver::isPaused()
	{
		if (m_Instance == nullptr) return false;
		return m_Instance->isPaused();
	}

	void Solver::updatePrefix(System::String ^prefix)
	{
		m_Instance->updatePrefix(ToNativeString(prefix));
	}

	void Solver::updateTarget(System::String ^target)
	{
		m_Instance->updateTarget(ToNativeString(target));
	}

	void Solver::updateDifficulty(System::String ^difficulty)
	{
		m_Instance->updateDifficulty(ToNativeString(difficulty));
	}

	void Solver::startFinding()
	{
		m_Instance->startFinding();
	}

	void Solver::stopFinding()
	{
		m_Instance->stopFinding();
	}

	void Solver::pauseFinding(bool pauseFinding)
	{
		m_Instance->pauseFinding(pauseFinding);
	}

	uint64_t Solver::getTotalHashRate()
	{
		if (m_Instance == nullptr) return 0ull;
		return m_Instance->getTotalHashRate();
	}

	uint64_t Solver::getHashRateByDevice(System::String ^platformName, int const deviceID)
	{
		if (m_Instance == nullptr) return 0ull;
		return m_Instance->getHashRateByDevice(ToNativeString(platformName), deviceID);
	}

	void Solver::OnGetWorkPosition(unsigned __int64 %workPosition)
	{
		OnGetWorkPositionHandler(workPosition);
	}

	void Solver::OnResetWorkPosition(unsigned __int64 %lastPosition)
	{
		OnResetWorkPositionHandler(lastPosition);
	}

	void Solver::OnIncrementWorkPosition(unsigned __int64 %lastPosition, unsigned __int64 increment)
	{
		OnIncrementWorkPositionHandler(lastPosition, increment);
	}

	void Solver::OnMessage(System::String ^platformName, int deviceID, System::String ^type, System::String ^message)
	{
		OnMessageHandler(platformName, deviceID, type, message);
	}

	void Solver::OnSolution(System::String ^digest, System::String ^address, System::String ^challenge, System::String ^difficulty, System::String ^target, System::String ^solution, bool isCustomDifficulty)
	{
		OnSolutionHandler(digest, address, challenge, difficulty, target, solution, isCustomDifficulty);
	}
}
