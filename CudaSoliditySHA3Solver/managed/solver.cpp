#include "solver.h"

namespace CudaSolver
{
	Solver::Solver(System::String ^maxDifficulty) : ManagedObject(new CUDASolver(ToNativeString(maxDifficulty)))
	{
		m_managedOnGetKingAddress = gcnew OnGetKingAddressDelegate(this, &Solver::OnGetKingAddress);
		System::IntPtr getKingAddressStubPtr = System::Runtime::InteropServices::Marshal::GetFunctionPointerForDelegate(m_managedOnGetKingAddress);
		CUDASolver::GetKingAddressCallback getKingAddressFnPtr = static_cast<CUDASolver::GetKingAddressCallback>(getKingAddressStubPtr.ToPointer());
		m_Instance->setGetKingAddressCallback(getKingAddressFnPtr);
		System::GC::KeepAlive(m_managedOnGetKingAddress);

		m_managedOnGetSolutionTemplate = gcnew OnGetSolutionTemplateDelegate(this, &Solver::OnGetSolutionTemplate);
		System::IntPtr getSolutionTemplateStubPtr = System::Runtime::InteropServices::Marshal::GetFunctionPointerForDelegate(m_managedOnGetSolutionTemplate);
		CUDASolver::GetSolutionTemplateCallback getSolutionTemplateFnPtr = static_cast<CUDASolver::GetSolutionTemplateCallback>(getSolutionTemplateStubPtr.ToPointer());
		m_Instance->setGetSolutionTemplateCallback(getSolutionTemplateFnPtr);
		System::GC::KeepAlive(m_managedOnGetSolutionTemplate);

		m_managedOnGetWorkPosition = gcnew OnGetWorkPositionDelegate(this, &Solver::OnGetWorkPosition);
		System::IntPtr getWorkPositionStubPtr = System::Runtime::InteropServices::Marshal::GetFunctionPointerForDelegate(m_managedOnGetWorkPosition);
		CUDASolver::GetWorkPositionCallback getWorkPositionFnPtr = static_cast<CUDASolver::GetWorkPositionCallback>(getWorkPositionStubPtr.ToPointer());
		m_Instance->setGetWorkPositionCallback(getWorkPositionFnPtr);
		System::GC::KeepAlive(m_managedOnGetWorkPosition);

		m_managedOnResetWorkPosition = gcnew OnResetWorkPositionDelegate(this, &Solver::OnResetWorkPosition);
		System::IntPtr resetWorkPositionStubPtr = System::Runtime::InteropServices::Marshal::GetFunctionPointerForDelegate(m_managedOnResetWorkPosition);
		CUDASolver::ResetWorkPositionCallback resetWorkPositionFnPtr = static_cast<CUDASolver::ResetWorkPositionCallback>(resetWorkPositionStubPtr.ToPointer());
		m_Instance->setResetWorkPositionCallback(resetWorkPositionFnPtr);
		System::GC::KeepAlive(m_managedOnResetWorkPosition);

		m_managedOnIncrementWorkPosition = gcnew OnIncrementWorkPositionDelegate(this, &Solver::OnIncrementWorkPosition);
		System::IntPtr incrementWorkPositionStubPtr = System::Runtime::InteropServices::Marshal::GetFunctionPointerForDelegate(m_managedOnIncrementWorkPosition);
		CUDASolver::IncrementWorkPositionCallback incrementWorkPositionFnPtr = static_cast<CUDASolver::IncrementWorkPositionCallback>(incrementWorkPositionStubPtr.ToPointer());
		m_Instance->setIncrementWorkPositionCallback(incrementWorkPositionFnPtr);
		System::GC::KeepAlive(m_managedOnIncrementWorkPosition);

		m_managedOnMessage = gcnew OnMessageDelegate(this, &Solver::OnMessage);
		System::IntPtr messageStubPtr = System::Runtime::InteropServices::Marshal::GetFunctionPointerForDelegate(m_managedOnMessage);
		CUDASolver::MessageCallback messageFnPtr = static_cast<CUDASolver::MessageCallback>(messageStubPtr.ToPointer());
		m_Instance->setMessageCallback(messageFnPtr);
		System::GC::KeepAlive(m_managedOnMessage);
		
		m_managedOnSolution = gcnew OnSolutionDelegate(this, &Solver::OnSolution);
		System::IntPtr solutionStubPtr = System::Runtime::InteropServices::Marshal::GetFunctionPointerForDelegate(m_managedOnSolution);
		CUDASolver::SolutionCallback solutionFnPtr = static_cast<CUDASolver::SolutionCallback>(solutionStubPtr.ToPointer());
		m_Instance->setSolutionCallback(solutionFnPtr);
		System::GC::KeepAlive(m_managedOnSolution);
	}

	Solver::~Solver()
	{
		try { m_Instance->~CUDASolver(); }
		catch(...) {}
	}

	void Solver::OnGetKingAddress(uint8_t *kingAddress)
	{
		OnGetKingAddressHandler(kingAddress);
	}

	bool Solver::foundNvAPI64()
	{
		return CUDASolver::foundNvAPI64();
	}

	int Solver::getDeviceCount(System::String ^%errorMessage)
	{
		std::string errMsg;
		int devCount = CUDASolver::getDeviceCount(errMsg);

		errorMessage = ToManagedString(errMsg);
		return devCount;
	}

	System::String ^Solver::getDeviceName(int deviceID, System::String ^%errorMessage)
	{
		std::string errMsg, devName;
		devName = CUDASolver::getDeviceName(deviceID, errMsg);

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

	bool Solver::assignDevice(int const deviceID, float const intensity)
	{
		return m_Instance->assignDevice(deviceID, intensity);
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

	uint64_t Solver::getHashRateByDeviceID(int const deviceID)
	{
		if (m_Instance == nullptr) return 0ull;
		return m_Instance->getHashRateByDeviceID(deviceID);
	}

	int Solver::getDeviceSettingMaxCoreClock(int deviceID)
	{
		return m_Instance->getDeviceSettingMaxCoreClock(deviceID);
	}

	int Solver::getDeviceSettingMaxMemoryClock(int deviceID)
	{
		return m_Instance->getDeviceSettingMaxMemoryClock(deviceID);
	}

	int Solver::getDeviceSettingPowerLimit(int deviceID)
	{
		return m_Instance->getDeviceSettingPowerLimit(deviceID);
	}

	int Solver::getDeviceSettingThermalLimit(int deviceID)
	{
		return m_Instance->getDeviceSettingThermalLimit(deviceID);
	}

	int Solver::getDeviceSettingFanLevelPercent(int deviceID)
	{
		return m_Instance->getDeviceSettingFanLevelPercent(deviceID);
	}

	int Solver::getDeviceCurrentFanTachometerRPM(int deviceID)
	{
		return m_Instance->getDeviceCurrentFanTachometerRPM(deviceID);
	}

	int Solver::getDeviceCurrentTemperature(int deviceID)
	{
		return m_Instance->getDeviceCurrentTemperature(deviceID);
	}

	int Solver::getDeviceCurrentCoreClock(int deviceID)
	{
		return m_Instance->getDeviceCurrentCoreClock(deviceID);
	}

	int Solver::getDeviceCurrentMemoryClock(int deviceID)
	{
		return m_Instance->getDeviceCurrentMemoryClock(deviceID);
	}

	int Solver::getDeviceCurrentUtilizationPercent(int deviceID)
	{
		return m_Instance->getDeviceCurrentUtilizationPercent(deviceID);
	}

	int Solver::getDeviceCurrentPstate(int deviceID)
	{
		return m_Instance->getDeviceCurrentPstate(deviceID);
	}

	System::String ^Solver::getDeviceCurrentThrottleReasons(int deviceID)
	{
		return ToManagedString(m_Instance->getDeviceCurrentThrottleReasons(deviceID));
	}

	void Solver::OnGetSolutionTemplate(uint8_t *solutionTemplate)
	{
		OnGetSolutionTemplateHandler(solutionTemplate);
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

	void Solver::OnMessage(int deviceID, System::String ^type, System::String ^message)
	{
		OnMessageHandler(deviceID, type, message);
	}

	void Solver::OnSolution(System::String ^digest, System::String ^address, System::String ^challenge, System::String ^difficulty, System::String ^target, System::String ^solution, bool isCustomDifficulty)
	{
		OnSolutionHandler(digest, address, challenge, difficulty, target, solution, isCustomDifficulty);
	}
}
