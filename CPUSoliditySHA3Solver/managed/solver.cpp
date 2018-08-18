#include "solver.h"

namespace CPUSolver
{
	Solver::Solver(System::String ^maxDifficulty, System::String ^threads) :
		ManagedObject(new cpuSolver(ToNativeString(maxDifficulty), ToNativeString(threads)))
	{
		m_managedOnGetKingAddress = gcnew OnGetKingAddressDelegate(this, &Solver::OnGetKingAddress);
		System::IntPtr getKingAddressStubPtr = System::Runtime::InteropServices::Marshal::GetFunctionPointerForDelegate(m_managedOnGetKingAddress);
		cpuSolver::GetKingAddressCallback getKingAddressFnPtr = static_cast<cpuSolver::GetKingAddressCallback>(getKingAddressStubPtr.ToPointer());
		m_Instance->setGetKingAddressCallback(getKingAddressFnPtr);
		System::GC::KeepAlive(m_managedOnGetKingAddress);

		m_managedOnGetWorkPosition = gcnew OnGetWorkPositionDelegate(this, &Solver::OnGetWorkPosition);
		System::IntPtr getWorkPositionStubPtr = System::Runtime::InteropServices::Marshal::GetFunctionPointerForDelegate(m_managedOnGetWorkPosition);
		cpuSolver::GetWorkPositionCallback getWorkPositionFnPtr = static_cast<cpuSolver::GetWorkPositionCallback>(getWorkPositionStubPtr.ToPointer());
		m_Instance->setGetWorkPositionCallback(getWorkPositionFnPtr);
		System::GC::KeepAlive(m_managedOnGetWorkPosition);

		m_managedOnResetWorkPosition = gcnew OnResetWorkPositionDelegate(this, &Solver::OnResetWorkPosition);
		System::IntPtr resetWorkPositionStubPtr = System::Runtime::InteropServices::Marshal::GetFunctionPointerForDelegate(m_managedOnResetWorkPosition);
		cpuSolver::ResetWorkPositionCallback resetWorkPositionFnPtr = static_cast<cpuSolver::ResetWorkPositionCallback>(resetWorkPositionStubPtr.ToPointer());
		m_Instance->setResetWorkPositionCallback(resetWorkPositionFnPtr);
		System::GC::KeepAlive(m_managedOnResetWorkPosition);

		m_managedOnIncrementWorkPosition = gcnew OnIncrementWorkPositionDelegate(this, &Solver::OnIncrementWorkPosition);
		System::IntPtr incrementWorkPositionStubPtr = System::Runtime::InteropServices::Marshal::GetFunctionPointerForDelegate(m_managedOnIncrementWorkPosition);
		cpuSolver::IncrementWorkPositionCallback incrementWorkPositionFnPtr = static_cast<cpuSolver::IncrementWorkPositionCallback>(incrementWorkPositionStubPtr.ToPointer());
		m_Instance->setIncrementWorkPositionCallback(incrementWorkPositionFnPtr);
		System::GC::KeepAlive(m_managedOnIncrementWorkPosition);

		m_managedOnGetSolutionTemplate = gcnew OnGetSolutionTemplateDelegate(this, &Solver::OnGetSolutionTemplate);
		System::IntPtr getSolutionTemplateStubPtr = System::Runtime::InteropServices::Marshal::GetFunctionPointerForDelegate(m_managedOnGetSolutionTemplate);
		cpuSolver::GetSolutionTemplateCallback getSolutionTemplateFnPtr = static_cast<cpuSolver::GetSolutionTemplateCallback>(getSolutionTemplateStubPtr.ToPointer());
		m_Instance->setGetSolutionTemplateCallback(getSolutionTemplateFnPtr);
		System::GC::KeepAlive(m_managedOnGetSolutionTemplate);

		m_managedOnMessage = gcnew OnMessageDelegate(this, &Solver::OnMessage);
		System::IntPtr messageStubPtr = System::Runtime::InteropServices::Marshal::GetFunctionPointerForDelegate(m_managedOnMessage);
		cpuSolver::MessageCallback messageFnPtr = static_cast<cpuSolver::MessageCallback>(messageStubPtr.ToPointer());
		m_Instance->setMessageCallback(messageFnPtr);
		System::GC::KeepAlive(m_managedOnMessage);
		
		m_managedOnSolution = gcnew OnSolutionDelegate(this, &Solver::OnSolution);
		System::IntPtr solutionStubPtr = System::Runtime::InteropServices::Marshal::GetFunctionPointerForDelegate(m_managedOnSolution);
		cpuSolver::SolutionCallback solutionFnPtr = static_cast<cpuSolver::SolutionCallback>(solutionStubPtr.ToPointer());
		m_Instance->setSolutionCallback(solutionFnPtr);
		System::GC::KeepAlive(m_managedOnSolution);
	}

	Solver::~Solver()
	{
		try { m_Instance->~cpuSolver(); }
		catch(...) {}
	}

	void Solver::OnGetKingAddress(uint8_t *kingAddress)
	{
		OnGetKingAddressHandler(kingAddress);
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

	unsigned int Solver::getLogicalProcessorsCount()
	{
		return cpuSolver::getLogicalProcessorsCount();
	}

	System::String ^Solver::getNewSolutionTemplate(System::String ^kingAddress)
	{
		return ToManagedString(cpuSolver::getNewSolutionTemplate(ToNativeString(kingAddress)));
	}

	void Solver::setCustomDifficulty(uint32_t customDifficulty)
	{
		m_Instance->setCustomDifficulty(customDifficulty);
	}

	void Solver::setSubmitStale(bool submitStale)
	{
		m_Instance->m_SubmitStale = submitStale;
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

	void Solver::updatePrefix(System::String^ prefix)
	{
		m_Instance->updatePrefix(ToNativeString(prefix));
	}

	void Solver::updateTarget(System::String^ target)
	{
		m_Instance->updateTarget(ToNativeString(target));
	}

	void Solver::updateDifficulty(System::String^ difficulty)
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

	uint64_t Solver::getHashRateByThreadID(unsigned int const threadID)
	{
		if (m_Instance == nullptr) return 0ull;
		return m_Instance->getHashRateByThreadID(threadID);
	}

	void Solver::OnMessage(int threadID, System::String^ type, System::String^ message)
	{
		OnMessageHandler(threadID, type, message);
	}

	void Solver::OnSolution(System::String^ digest, System::String^ address, System::String^ challenge, System::String^ difficulty, System::String^ target, System::String^ solution, bool isCustomDifficulty)
	{
		OnSolutionHandler(digest, address, challenge, difficulty, target, solution, isCustomDifficulty);
	}
}
