#include "solver.h"

namespace CPUSolver
{
	Solver::Solver(System::String ^maxDifficulty, System::String ^threads, System::String ^solutionTemplate, System::String ^kingAddress) :
		ManagedObject(new cpuSolver(ToNativeString(maxDifficulty), ToNativeString(threads), ToNativeString(solutionTemplate), ToNativeString(kingAddress)))
	{
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

	unsigned int Solver::getLogicalProcessorsCount()
	{
		return cpuSolver::getLogicalProcessorsCount();
	}

	System::String ^Solver::getSolutionTemplate(System::String ^kingAddress)
	{
		return ToManagedString(cpuSolver::getSolutionTemplate(ToNativeString(kingAddress)));
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
		return m_Instance->isMining();
	}

	bool Solver::isPaused()
	{
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
		return m_Instance->getTotalHashRate();
	}

	uint64_t Solver::getHashRateByThreadID(unsigned int const threadID)
	{
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
